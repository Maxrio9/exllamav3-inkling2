// Fused decode-shaped MoE path (bsz 1..MAX_BSZN): launcher and tensor front end. Kernels in
// exl3_moe_coop_kernel.cuh, instantiated per (K, codebook) in comp_units/exl3_moe_coop_inst_k*.cu

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <cmath>

#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_coop.cuh"
#include "comp_units/exl3_moe_coop_instances.cuh"
#include "bits_k.cuh"
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exl3_moe_coop_kernel.cuh"
#ifndef USE_ROCM
#include "exl3_moe_coop_inkling.cuh"
#endif

// Output rows of a launch without routed slots (top-k 0): the empty sum, plus the shared-expert term
// when there is one. One warp per (row, 128-column chunk)
__global__ __launch_bounds__(32)
void exl3_moe_coop_empty_kernel(const MoeCoopParams p)
{
    exl3_moe_coop_ns::write_empty_row_chunk(p, blockIdx.x, blockIdx.y, threadIdx.x);
}

static MoeCoopKernel moe_coop_kernel_a(float K_, int cb, int Hi, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_a_h1(Hi, wide);
            case 2: return exl3_moe_coop_kernel_a_h2(Hi, wide);
            default: return exl3_moe_coop_kernel_a_h3(Hi, wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_a_k1(cb, Hi, wide);
        case 2: return exl3_moe_coop_kernel_a_k2(cb, Hi, wide);
        case 3: return exl3_moe_coop_kernel_a_k3(cb, Hi, wide);
        case 4: return exl3_moe_coop_kernel_a_k4(cb, Hi, wide);
        case 5: return exl3_moe_coop_kernel_a_k5(cb, Hi, wide);
        case 6: return exl3_moe_coop_kernel_a_k6(cb, Hi, wide);
        case 7: return exl3_moe_coop_kernel_a_k7(cb, Hi, wide);
        default: return exl3_moe_coop_kernel_a_k8(cb, Hi, wide);
    }
}

static MoeCoopKernel moe_coop_kernel_b(float K_, int cb, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_b_h1(wide);
            case 2: return exl3_moe_coop_kernel_b_h2(wide);
            default: return exl3_moe_coop_kernel_b_h3(wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_b_k1(cb, wide);
        case 2: return exl3_moe_coop_kernel_b_k2(cb, wide);
        case 3: return exl3_moe_coop_kernel_b_k3(cb, wide);
        case 4: return exl3_moe_coop_kernel_b_k4(cb, wide);
        case 5: return exl3_moe_coop_kernel_b_k5(cb, wide);
        case 6: return exl3_moe_coop_kernel_b_k6(cb, wide);
        case 7: return exl3_moe_coop_kernel_b_k7(cb, wide);
        default: return exl3_moe_coop_kernel_b_k8(cb, wide);
    }
}

// Split-k factor: blocks per column chunk, partial outputs summed by the last-arriving block.
// Measured ineffective.
// Split-k factor for one stage. The grid is slots x column tiles, which fills the device for a
// routed launch (bsz x top-k slots) but not for a single-slot launch such as the shared expert
// (one expert's tiles on a 100+ SM device): split k until there is a block per SM, keeping at
// least 64 k-slices (1024 columns) per split so the partials and their reduction stay cheap.
// EXL3_MOE_COOP_KSPLIT forces one factor for every stage (testing)
static int moe_coop_pick_ksplit(int base, int kslices, int max_split, int device)
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_KSPLIT");
        forced = env ? atoi(env) : 0;
    }
    if (forced > 0) return std::min(forced, max_split);
    const int want = CEIL_DIVIDE(DevCtx::instance().get_num_sms(device), std::max(base, 1));
    const int kcap = std::max(kslices / 64, 1);
    return std::max(1, std::min(std::min(want, kcap), max_split));
}

// Tile geometry per stage: the wide (128-column, 4-way k-split) tile for long k, the narrow one
// otherwise. EXL3_MOE_COOP_WIDE = 0 / 1 forces one geometry for every stage (testing)
static int moe_coop_wide_mode()
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_WIDE");
        mode = env ? atoi(env) : -1;
    }
    return mode;
}

static bool moe_coop_pick_wide(int kslices, int slots, int device)
{
    const int mode = moe_coop_wide_mode();
    if (mode >= 0) return mode != 0;
    if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}

// Dynamic shared memory above the 48 KB default needs an opt-in per kernel (once per process)
static void moe_coop_smem_optin(void* kernel, int smem)
{
    if (smem <= 48 * 1024) return;
    static std::map<void*, int> done;
    auto it = done.find(kernel);
    if (it != done.end() && it->second >= smem) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    done[kernel] = smem;
}

void exl3_moe_coop_launch(const MoeCoopParams& p_in, float K_gu, float K_d, int cb, int device, cudaStream_t stream)
{
    MoeCoopParams p = p_in;
    { static int dbg = std::getenv("EXL3_MOE_COOP_DBG") ? atoi(std::getenv("EXL3_MOE_COOP_DBG")) : 0; p.dbg = dbg; }
    const int slots = p.bsz * p.topk;
    if (!slots)
    {
        if (p.bsz && p.H_out)
        {
            exl3_moe_coop_empty_kernel<<<dim3(p.bsz, CEIL_DIVIDE(p.H_out, 128)), 32, 0, stream>>>(p);
            cuda_check(cudaPeekAtLastError());
        }
        return;
    }
    const int nproj = p.gated ? 2 : 1;
    p.a_global = p.bsz > 1 && !p.gu_k2;

    const bool wide_a = !p.gu_k2 && moe_coop_pick_wide(p.Hi / 16, slots, device);
    const bool wide_b = !p.gu_k2 && moe_coop_pick_wide(p.I / 16, slots, device);
    using namespace exl3_moe_coop_ns;
    MoeCoopKernel ka, kb;
    if (p.gu_k2)
    {
        // Use per-slot tiles for small verification blocks. The CTA rotates
        // its input locally, so the packed path always takes two launches.
        ka = {(void*)exl3_moe_coop_a_kernel<0, 2, false>, smem_a_bytes<0>(p.Hi * (p.slot_tiles ? ROWS : 1))};
        kb = {(void*)exl3_moe_coop_b_kernel<0, 2, false>, smem_b_bytes<0>()};
    }
    else
    {
        ka = moe_coop_kernel_a(K_gu, cb, p.Hi, wide_a);
        kb = moe_coop_kernel_b(K_d, cb, wide_b);
    }
    moe_coop_smem_optin(ka.kernel, ka.smem);
    moe_coop_smem_optin(kb.kernel, kb.smem);

    const int runs = p.slot_tiles ? p.num_tiles : slots;
    const int base_a = runs * nproj * (p.I / (wide_a ? 128 : MOE_COOP_COLS));
    const int base_b = runs * (p.Ho / (wide_b ? 128 : MOE_COOP_COLS));
    const int max_split = std::max(1, p.slots_max / slots);
    p.ksplit_a = moe_coop_pick_ksplit(base_a, p.Hi / 16, max_split, device);
    p.ksplit_b = moe_coop_pick_ksplit(base_b, p.I / 16, max_split, device);
    const int grid_a = base_a * p.ksplit_a;
    const int grid_b = base_b * p.ksplit_b;

    void* args[] = { (void*) &p };
    if (p.a_global)
    {
        const int rot_items = slots * (p.Hi / 128) * nproj;
        const int rot_grid = CEIL_DIVIDE(rot_items, MOE_COOP_THREADS / 32);
        exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<rot_grid, MOE_COOP_THREADS, 0, stream>>>(p);
    }
    cuda_check(cudaLaunchKernel(ka.kernel, dim3(grid_a), dim3(MOE_COOP_THREADS), args, ka.smem, stream));
    cuda_check(cudaLaunchKernel(kb.kernel, dim3(grid_b), dim3(MOE_COOP_THREADS), args, kb.smem, stream));
    cuda_check(cudaPeekAtLastError());
}

MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
)
{
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(ctr.is_contiguous(), "exl3_moe_coop: ctr must be contiguous");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous(), "exl3_moe_coop: pointer tables must be contiguous int64");
    TORCH_CHECK(!(mcg && mul1), "exl3_moe_coop: specified both mcg and mul1");
    TORCH_CHECK(Kg >= 1 && Kg <= 8 && Ku >= 1 && Ku <= 8 && Kd >= 1 && Kd <= 8, "exl3_moe_coop: K out of range");
    TORCH_CHECK(!gated || Kg == Ku, "exl3_moe_coop: gate and up must share a bit width");
    TORCH_CHECK(act >= 0 && act <= 3, "exl3_moe_coop: unknown activation");

    MoeCoopParams p = {};
    p.Hi = Hi;
    p.I = (int) gu_u.size(-1);
    p.Ho = (int) d_out.size(-1);
    p.H_out = (int) out.size(-1);
    TORCH_CHECK(p.Hi % 128 == 0 && p.I % 128 == 0 && p.Ho % 128 == 0 && p.H_out <= p.Ho, "exl3_moe_coop: Hi/I/Ho shape");
    TORCH_CHECK(p.Hi > 0 && p.I > 0 && p.Ho > 0, "exl3_moe_coop: empty Hi/I/Ho dimension");

    TORCH_CHECK_DTYPE(had_g, kHalf);
    TORCH_CHECK_DTYPE(had_u, kHalf);
    TORCH_CHECK_DTYPE(act_out, kHalf);
    TORCH_CHECK_DTYPE(d_out, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    p.gu_f32 = gu_u.scalar_type() == at::kFloat;
    if (!p.gu_f32) TORCH_CHECK_DTYPE(gu_u, kHalf);
    TORCH_CHECK(gu_g.scalar_type() == gu_u.scalar_type(), "exl3_moe_coop: gate/up scratch dtype mismatch");
    // Scratch capacity in slots: the smallest of the per-slot buffers
    const int slots_max = (int) std::min({ had_g.numel() / p.Hi, had_u.numel() / p.Hi, gu_g.numel() / p.I,
                                           gu_u.numel() / p.I, act_out.numel() / p.I, d_out.numel() / p.Ho });
    auto check_scratch = [&] (const at::Tensor& t, int width)
    {
        TORCH_CHECK(t.is_contiguous() && t.size(-1) == width, "exl3_moe_coop: scratch shape");
    };
    check_scratch(had_g, p.Hi);
    check_scratch(had_u, p.Hi);
    check_scratch(gu_g, p.I);
    check_scratch(gu_u, p.I);
    check_scratch(act_out, p.I);
    check_scratch(d_out, p.Ho);
    TORCH_CHECK(out.dim() >= 2 && out.stride(-1) == 1, "exl3_moe_coop: out shape");
    TORCH_CHECK(slots_max >= 1 && slots_max <= 256, "exl3_moe_coop: scratch must hold 1..256 slots");

    // Counters: the gate/up stage's block covers every slot the scratch can hold, the down
    // stage's every token row the output can hold
    const int rows_max = (int) out.size(-2);
    p.slots_max = slots_max;
    p.ctr_a_len = slots_max * (p.I / 128);
    p.ctr_b_len = rows_max * (p.Ho / 128);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop_ctr_len(slots_max, rows_max, p.I, p.Ho), "exl3_moe_coop: counter scratch too small");
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.rows_max = rows_max;

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.g_bias = g_bias ? (const int64_t*) g_bias->data_ptr() : nullptr;
    p.u_bias = u_bias ? (const int64_t*) u_bias->data_ptr() : nullptr;
    p.d_bias = d_bias ? (const int64_t*) d_bias->data_ptr() : nullptr;
    p.n_local = (int) u_trellis.size(0);
    p.act = act;
    p.act_limit = act_limit;
    p.gated = gated;

    p.had_g = (half*) had_g.data_ptr();
    p.had_u = (half*) had_u.data_ptr();
    p.gu_g = gu_g.data_ptr();
    p.gu_u = gu_u.data_ptr();
    p.act_out = (half*) act_out.data_ptr();
    p.d_out = (float*) d_out.data_ptr();
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(-2);

    if (sh_gate_w)
    {
        TORCH_CHECK_DTYPE(sh_gate_w.value(), kHalf);
        TORCH_CHECK(sh_gate_w->is_contiguous(), "exl3_moe_coop: shared gate weight must be contiguous");
        p.sh_gate_w = (const half*) sh_gate_w->data_ptr();
        p.sh_gate_n = (int) sh_gate_w->numel();
    }

    K_gu = gated ? Kg : Ku;
    K_d = Kd;
    cb = mcg ? 1 : (mul1 ? 2 : 0);
    return p;
}

void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    // Per-call checks kept to what can actually vary between calls
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "exl3_moe_coop: x must be (bsz, H) with unit column stride");
    p.x = (const half*) x.data_ptr();
    p.x_stride = (int) x.stride(0);
    p.bsz = (int) x.size(0);
    p.H = (int) x.size(1);
    p.topk = (int) sel.size(-1);
    p.sel = (const int64_t*) sel.data_ptr();
    p.rw = (const half*) rw.data_ptr();
    const int slots = p.bsz * p.topk;
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi, "exl3_moe_coop: H/Hi shape");
    TORCH_CHECK(slots <= p.slots_max && p.bsz <= p.rows_max, "exl3_moe_coop: batch exceeds the scratch");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop: expert range exceeds tables");
    if (sh_out)
    {
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
        if (p.sh_gate_w) TORCH_CHECK(p.sh_gate_n == p.H && p.H % 2 == 0, "exl3_moe_coop: shared gate width");
    }
    else
        p.sh_gate_w = nullptr;
    exl3_moe_coop_launch(p, K_gu, K_d, cb, device, stream);
}

void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(rw, kHalf);
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous(), "exl3_moe_coop: sel/rw must be contiguous");
    TORCH_CHECK(sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0), "exl3_moe_coop: sel/rw must be (bsz, topk)");
    if (sh_out)
    {
        TORCH_CHECK_DTYPE(sh_out.value(), kFloat);
        TORCH_CHECK(sh_out->is_contiguous() && sh_out->size(-1) == x.size(1) && sh_out->numel() >= x.size(0) * x.size(1),
                    "exl3_moe_coop: shared expert output shape");
    }
    float K_gu, K_d; int cb;
    MoeCoopParams p = exl3_moe_coop_prepare(Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh, d_trellis, d_suh, d_svh,
                                            g_bias, u_bias, d_bias, Kg, Ku, Kd, mcg, mul1, act, act_limit, gated,
                                            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, sh_gate_w, K_gu, K_d, cb);
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);
}

namespace exl3_moe_coop_ns {

// Packed gate/up shares an input rotation. Store it once per selected slot in
// the down scratch, whose first consumer is after the gate/up GEMM completes.
__global__ void inkling_rotate(const MoeCoopParams p)
{
    __shared__ int16_t order[MAX_SLOTS], experts[MAX_SLOTS];
    __shared__ int scan[WK], result[4];
    if (p.runs && !p.slot_tiles && blockIdx.x == 0)
        build_runs(p, order, experts, scan, result);
    const int item = blockIdx.x * WK + threadIdx.x / 32;
    const int chunks = p.H / 128;
    const int slot = item / chunks;
    if (slot >= p.slots_max) return;
    rotate_chunk(p, slot / 8, (const half*) p.g_suh[p.sel[slot]], item % chunks,
                 p.had_g + (size_t) slot * p.H, threadIdx.x % 32);
}

// Use the same k splits, fragment folding and reduction order as the decode
// kernel. A stream boundary replaces the per-slot completion counters.
template <bool DOWN, int TILE_ROWS>
__global__ __launch_bounds__(THREADS)
void inkling_grouped_gemm(const MoeCoopParams p)
{
    __shared__ int slots[TILE_ROWS];
    __shared__ float red[WK * TILE_ROWS * COLS];
    __shared__ uint32_t stage[WK * STAGE_WORDS];
    const int run = blockIdx.y;
    int nrows;
    if constexpr (TILE_ROWS == 16)
    {
        const int* tile = p.slot_tiles + run * TILE_ROWS;
        if (threadIdx.x < TILE_ROWS) slots[threadIdx.x] = tile[threadIdx.x];
        __syncthreads();
        if (slots[0] < 0) return;
        nrows = __popc(__ballot_sync(0xffffffffu, (threadIdx.x % 32 < TILE_ROWS) &&
                                    slots[threadIdx.x % TILE_ROWS] >= 0));
    }
    else if (!read_run(p, run, slots, nrows, nullptr)) return;
    const int expert = p.sel[slots[0]];
    const int group = blockIdx.x;
    const int cols = DOWN ? p.H : p.I;
    const int inner = DOWN ? p.I : p.H;
    const auto* rates = DOWN ? p.down_k2 : p.gu_k2;
    auto* packed = (const uint32_t*) (DOWN ? p.d_trellis : p.g_trellis)[expert];
    const bool up = !DOWN && blockIdx.z == 1;
    if (up) packed += (p.I / 16) * (4 * rates[expert]);
    auto* input = (const half2*) (DOWN ? p.act_out : p.had_g);
    void* output = DOWN ? p.d_out : (up ? p.gu_u : p.gu_g);
    routed_tile<0, 2, false, false, TILE_ROWS>(
        rates, expert, packed, input, (size_t) inner / 2, slots, nrows, output, (size_t) cols,
        true, 0, inner / 16, (DOWN ? cols : 2 * cols) / 16, group, red, stage);
}

__device__ __forceinline__ float4 inkling_output_chunk(
    const float* raw, const half* scale, int lane)
{
    float4 v = *reinterpret_cast<const float4*>(raw);
    had128(v.x, v.y, v.z, v.w, lane);
    scale_h4(scale, v.x, v.y, v.z, v.w);
    return v;
}

__global__ void inkling_activate(const MoeCoopParams p)
{
    const int item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int slot = item / (p.I / 128);
    if (slot >= p.slots_max) return;
    const int lane = threadIdx.x % 32, expert = p.sel[slot];
    const int col = item % (p.I / 128) * 128 + lane * 4;
    const int off = slot * p.I + col;
    auto* svh = (const half*) p.g_svh[expert] + col;
    float4 g = inkling_output_chunk((const float*) p.gu_g + off, svh, lane);
    float4 u = inkling_output_chunk((const float*) p.gu_u + off, svh + p.I, lane);
    float a = inkling_swiglu(g.x, u.x, p.act_limit, p.up_cap);
    float b = inkling_swiglu(g.y, u.y, p.act_limit, p.up_cap);
    float c = inkling_swiglu(g.z, u.z, p.act_limit, p.up_cap);
    float d = inkling_swiglu(g.w, u.w, p.act_limit, p.up_cap);
    scale_h4((const half*) p.d_suh[expert] + col, a, b, c, d);
    had128(a, b, c, d, lane);
    store_h4(p.act_out + off, a, b, c, d);
}

__global__ void inkling_reduce(const MoeCoopParams p)
{
    __shared__ float4 weighted[8][32];
    const int lane = threadIdx.x % 32, pick = threadIdx.x / 32;
    const int slot = blockIdx.y * 8 + pick, col = blockIdx.x * 128 + lane * 4;
    float4 v = inkling_output_chunk(p.d_out + slot * p.H + col,
                                    (const half*) p.d_svh[p.sel[slot]] + col, lane);
    const float w = p.rw_float[slot];
    weighted[pick][lane] = make_float4(w * __half2float(__float2half_rn(v.x)),
                                       w * __half2float(__float2half_rn(v.y)),
                                       w * __half2float(__float2half_rn(v.z)),
                                       w * __half2float(__float2half_rn(v.w)));
    __syncthreads();
    if (!pick)
    {
        float4 sum = {};
        #pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            float4 r = weighted[k][lane];
            sum.x += r.x; sum.y += r.y; sum.z += r.z; sum.w += r.w;
        }
        *reinterpret_cast<float4*>(p.out + blockIdx.y * p.H + col) = sum;
    }
}

template <int TILE_ROWS>
static void inkling_batch(MoeCoopParams p, cudaStream_t stream)
{
    const int tiles = p.slot_tiles ? p.num_tiles : p.slots_max;
    p.a_global = true;
    p.had_g = reinterpret_cast<half*>(p.d_out);
    inkling_rotate<<<CEIL_DIVIDE(p.slots_max * (p.H / 128), WK), THREADS, 0, stream>>>(p);
    inkling_grouped_gemm<false, TILE_ROWS><<<dim3(p.I / COLS, tiles, 2), THREADS, 0, stream>>>(p);
    inkling_activate<<<CEIL_DIVIDE(p.slots_max * (p.I / 128), 8), 256, 0, stream>>>(p);
    inkling_grouped_gemm<true, TILE_ROWS><<<dim3(p.H / COLS, tiles), THREADS, 0, stream>>>(p);
    inkling_reduce<<<dim3(p.H / 128, p.bsz), 256, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
}

#ifndef USE_ROCM
template <bool SINGLE>
static void inkling_decode(MoeCoopParams p, cudaStream_t stream)
{
    p.a_global = true;
    p.had_g = reinterpret_cast<half*>(p.d_out);
    inkling_rotate<<<CEIL_DIVIDE(p.slots_max * (p.H / 128), WK), THREADS, 0, stream>>>(p);
    const dim3 grid(INKLING_HIDDEN / 16, p.slots_max);
    inkling_decode_gemm<false, SINGLE><<<grid, INKLING_DECODE_WARPS * 32, 0, stream>>>(p);
    inkling_activate<<<CEIL_DIVIDE(p.slots_max * (p.I / 128), 8), 256, 0, stream>>>(p);
    inkling_decode_gemm<true, SINGLE><<<grid, INKLING_DECODE_WARPS * 32, 0, stream>>>(p);
    inkling_reduce<<<dim3(p.H / 128, p.bsz), 256, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
}
#endif
} // namespace exl3_moe_coop_ns

void inkling_moe_coop(const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
                      const at::Tensor& tables, const c10::optional<at::Tensor>& tiles, double gate_cap, double up_cap,
                      at::Tensor gu, at::Tensor activation, at::Tensor down, at::Tensor ctr, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) > 0 && x.size(0) <= (tiles ? 2048 : 8), "Inkling MoE input");
    for (const auto* t : {&sel, &rw, &tables, &x})
        TORCH_CHECK(t->device() == x.device() && t->is_contiguous(), "Inkling MoE input device/stride");
    for (const auto* t : {&gu, &activation, &down, &ctr, &out})
        TORCH_CHECK(t->device() == x.device() && t->is_contiguous(), "Inkling MoE scratch device/stride");
    TORCH_CHECK(tables.dim() == 3 && tables.size(0) == 2 && tables.size(1) == 4 && tables.size(2) >= 8,
                "Inkling MoE table layout");
    TORCH_CHECK(sel.dim() == 2 && sel.size(0) == x.size(0) && sel.size(1) == 8 && rw.sizes() == sel.sizes(),
                "Inkling MoE routing shape");
    const int slots = sel.numel(), H = x.size(1);
    TORCH_CHECK(activation.dim() == 2 && activation.size(0) == slots, "Inkling MoE activation shape");
    const int I = activation.size(1);
    TORCH_CHECK(H > 0 && I > 0 && H % 128 == 0 && I % 128 == 0, "Inkling MoE alignment");
    TORCH_CHECK(gu.sizes() == at::IntArrayRef({2, slots, I}) && down.sizes() == at::IntArrayRef({slots, H})
                && out.sizes() == x.sizes() && ctr.numel() >= slots * (I / 128) + x.size(0) * (H / 128),
                "Inkling MoE scratch shape");
    MoeCoopParams p = {};
    if (tiles)
    {
        TORCH_CHECK(tiles->device() == x.device() && tiles->is_contiguous() && tiles->dim() == 2
                    && (tiles->size(1) == 8 || tiles->size(1) == 16) && tiles->size(0) > 0 && tiles->size(0) <= 65535, "Inkling MoE slot tiles");
        p.slot_tiles = tiles->data_ptr<int>();
        p.num_tiles = tiles->size(0);
    }
    const int E = tables.size(2);
    const int64_t* fields = tables.data_ptr<int64_t>();
    p.g_trellis = p.u_trellis = fields;
    p.g_suh = p.u_suh = fields + E;
    p.g_svh = p.u_svh = fields + 2 * E;
    p.gu_k2 = fields + 3 * E;
    p.d_trellis = fields + 4 * E;
    p.d_suh = fields + 5 * E;
    p.d_svh = fields + 6 * E;
    p.down_k2 = fields + 7 * E;
    p.x = (const half*)x.data_ptr<at::Half>();
    p.sel = sel.data_ptr<int64_t>();
    p.rw_float = rw.data_ptr<float>();
    p.Hi = p.Ho = p.H = p.H_out = p.out_stride = p.x_stride = H;
    p.I = I;
    p.bsz = p.rows_max = x.size(0);
    p.topk = 8;
    p.slots_max = slots;
    p.min_expert = -1;
    p.gated = true;
    p.act_limit = gate_cap;
    p.up_cap = up_cap;
    p.gu_f32 = true;
    p.gu_g = gu.data_ptr<float>();
    p.gu_u = gu.data_ptr<float>() + slots * I;
    p.act_out = (half*)activation.data_ptr<at::Half>();
    p.d_out = down.data_ptr<float>();
    p.out = out.data_ptr<float>();
    p.ctr_a_len = slots * (I / 128);
    p.ctr_b_len = p.bsz * (H / 128);
    p.ctr_a = ctr.data_ptr<int>();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int counters = p.ctr_a_len + p.ctr_b_len;
    if (!tiles && p.bsz > 1 && E <= 32768 && ctr.numel() >= counters + 3 + 2 * slots)
        p.runs = ctr.data_ptr<int>() + counters;

#ifndef USE_ROCM
    if (!tiles && H == exl3_moe_coop_ns::INKLING_HIDDEN && I == exl3_moe_coop_ns::INKLING_INTER
        && (p.bsz == 1 || p.runs))
    {
        const auto* props = at::cuda::getDeviceProperties(x.get_device());
        if (props->major == 12 && props->minor == 1)
        {
            if (p.bsz == 1) exl3_moe_coop_ns::inkling_decode<true>(p, stream);
            else exl3_moe_coop_ns::inkling_decode<false>(p, stream);
            return;
        }
    }
#endif
    if (tiles && tiles->size(1) == 16)
        exl3_moe_coop_ns::inkling_batch<16>(p, stream);
    else if (p.runs)
        exl3_moe_coop_ns::inkling_batch<8>(p, stream);
    else
        exl3_moe_coop_launch(p, 0, 0, 2, x.get_device(), stream);
}
