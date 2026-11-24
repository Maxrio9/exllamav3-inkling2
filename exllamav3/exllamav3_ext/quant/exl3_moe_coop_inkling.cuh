#pragma once

namespace exl3_moe_coop_ns {

constexpr int INKLING_HIDDEN = 2304;
constexpr int INKLING_INTER = 1152;
constexpr int INKLING_DECODE_WARPS = 8;

// Each warp owns a 16-column MMA tile and one of the original 16 logical K
// partitions at a time. A whole partition is just 9 (gate/up) or 5 (down)
// trellis tiles. Issue all its packed reads before decoding the first tile.
template <int RATE>
__device__ __forceinline__ void inkling_fragment(uint32_t word, uint32_t second, int lane,
                                                 FragB& lo, FragB& hi)
{
    if constexpr (RATE == 6)
    {
        // A lane decodes eight consecutive windows. Source lanes wrap in the
        // 768-bit tile, including the history preceding the first window.
        int end = (lane + 1) * 24;
        int first = (end + 768 - 37) / 32 % 24;
        int last = (end - 1) / 32;
        exl3_gemv_ns::dq8_regs_3bits<2>(__shfl_sync(0xffffffff, word, first),
                                         __shfl_sync(0xffffffff, word, last), (-end) & 31, lo, hi);
    }
    else if constexpr (RATE == 7)
    {
        int end = (lane + 1) * 28;
        // The two four-window groups end 14 bits apart.
        int mid = end - 14;
        auto shuffle = [&](int bit) { return __shfl_sync(0xffffffff, word, ((bit + 896) / 32) % 28); };
        exl3_gemv_ns::dq8_regs_half<3, 2>(shuffle(end - 27), shuffle(end - 1), (-end) & 31,
                                          shuffle(mid - 27), shuffle(mid - 1), (-mid) & 31, lo, hi);
    }
    else if constexpr (RATE == 8)
    {
        exl3_gemv_ns::dq8_regs_4bits<2>(__shfl_sync(0xffffffff, word, (lane + 31) & 31), word, lo, hi);
    }
    else
    {
        // At 8 bpw each lane holds eight new bytes. The first index also
        // needs the preceding lane's last byte, wrapping within the tile.
        uint32_t previous = __shfl_sync(0xffffffff, second, (lane + 31) & 31);
        exl3_gemv_ns::decode8<2>(__funnelshift_r(word, previous, 24) & 65535, word >> 16,
                                 (word >> 8) & 65535, word & 65535,
                                 __funnelshift_r(second, word, 24) & 65535, second >> 16,
                                 (second >> 8) & 65535, second & 65535, lo, hi);
    }
}

template <int RATE, bool DOWN, bool SINGLE>
__device__ __forceinline__ void inkling_dot(const MoeCoopParams& p, const int* slots, int nrows,
                                            const uint32_t* packed, int tile, float2* red)
{
    constexpr int K = DOWN ? INKLING_INTER : INKLING_HIDDEN;
    // Preserve the original partition and four-MMA fold order for exact rounding.
    constexpr int PART = (K / 16 + 15) / 16;
    constexpr int ROW_COUNT = SINGLE ? 1 : 8;
    constexpr int WORDS = RATE * 4;
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    int row = lane / 4;
    int slot = SINGLE ? blockIdx.y : slots[row < nrows ? row : 0];
    const half2* input = (const half2*)(DOWN ? p.act_out : p.had_g) + slot * (K / 2) + (lane & 3);
    #pragma unroll
    for (int batch = 0; batch < 16 / INKLING_DECODE_WARPS; ++batch)
    {
        int part = batch * INKLING_DECODE_WARPS + warp;
        int begin = part * PART;
        uint32_t weights[PART], second[PART];
        #pragma unroll
        for (int i = 0; i < PART; ++i)
        {
            auto ptr = packed + ((begin + i) * (INKLING_HIDDEN / 16) + tile) * WORDS;
            bool valid = !DOWN || begin + i < K / 16;
            if constexpr (RATE == 16)
            {
                uint2 pair = valid ? __ldg((const uint2*)ptr + lane) : make_uint2(0, 0);
                weights[i] = pair.x;
                second[i] = pair.y;
            }
            else
            {
                weights[i] = valid && lane < WORDS ? __ldg(ptr + lane) : 0;
                second[i] = 0;
            }
        }

        FragC_h c0 = {}, c1 = {};
        float2 acc0 = {}, acc1 = {};
        #pragma unroll
        for (int i = 0; i < PART; ++i)
        {
            if (DOWN && begin + i >= K / 16) break;
            FragB lo, hi, a01 = {}, a23 = {};
            inkling_fragment<RATE>(weights[i], second[i], lane, lo, hi);
            if (row < nrows)
            {
                a01[0] = input[(begin + i) * 8];
                a23[0] = input[(begin + i) * 8 + 4];
            }
            exl3_gemv_ns::mma_ab_h(a01, a23, lo, c0);
            exl3_gemv_ns::mma_ab_h(a01, a23, hi, c1);
            if ((i + 1) % 4 == 0 || i + 1 == PART || (DOWN && begin + i + 1 == K / 16))
            {
                acc0.x += __low2float(c0[0]); acc0.y += __high2float(c0[0]);
                acc1.x += __low2float(c1[0]); acc1.y += __high2float(c1[0]);
                c0 = {}; c1 = {};
            }
        }
        if (row < nrows)
        {
            int base = (part * ROW_COUNT + row) * 8 + (lane & 3);
            red[base] = acc0;
            red[base + 4] = acc1;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nrows * 8; i += INKLING_DECODE_WARPS * 32)
    {
        float2 v = {};
        #pragma unroll
        for (int part = 0; part < 16; ++part)
        {
            float2 s = red[part * ROW_COUNT * 8 + i];
            v.x += s.x; v.y += s.y;
        }
        int dst_slot = SINGLE ? blockIdx.y : slots[i / 8];
        int col = blockIdx.x * 16 + 2 * (i % 8);
        float* output = p.d_out;
        if constexpr (!DOWN)
        {
            output = (float*)(col < INKLING_INTER ? p.gu_g : p.gu_u);
            col %= INKLING_INTER;
        }
        *reinterpret_cast<float2*>(output + dst_slot * (DOWN ? INKLING_HIDDEN : INKLING_INTER) + col) = v;
    }
}

template <bool DOWN, bool SINGLE>
__global__ __launch_bounds__(INKLING_DECODE_WARPS * 32)
void inkling_decode_gemm(const MoeCoopParams p)
{
    constexpr int ROW_COUNT = SINGLE ? 1 : 8;
    __shared__ int slots[ROW_COUNT];
    __shared__ float2 red[16 * ROW_COUNT * 8];
    int run = blockIdx.y;
    int nrows = 1;
    if constexpr (!SINGLE)
    {
        if (!read_run(p, run, slots, nrows, nullptr)) return;
    }
    int expert = p.sel[SINGLE ? run : slots[0]];
    int rate = (DOWN ? p.down_k2 : p.gu_k2)[expert];
    const auto* packed = (const uint32_t*)(DOWN ? p.d_trellis : p.g_trellis)[expert];
    int tile = blockIdx.x;
    switch (rate)
    {
        case 6: inkling_dot<6, DOWN, SINGLE>(p, slots, nrows, packed, tile, red); break;
        case 7: inkling_dot<7, DOWN, SINGLE>(p, slots, nrows, packed, tile, red); break;
        case 8: inkling_dot<8, DOWN, SINGLE>(p, slots, nrows, packed, tile, red); break;
        case 16: inkling_dot<16, DOWN, SINGLE>(p, slots, nrows, packed, tile, red); break;
    }
}
} // namespace exl3_moe_coop_ns
