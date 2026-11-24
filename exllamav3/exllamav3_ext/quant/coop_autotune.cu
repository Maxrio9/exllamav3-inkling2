#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "coop_autotune.cuh"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <vector>

#include "../util.cuh"
#include "../util.h"

//#define CACHEDEBUG 1

namespace
{

constexpr uint64_t COOP_AUTOTUNE_VERSION = 5;
constexpr char DISK_CACHE_MAGIC[8] = { 'E', 'X', '3', 'A', 'T', 'U', 'N', 'E' };
constexpr uint32_t DISK_CACHE_FORMAT = 1;

struct ExpandedCandidate
{
    void* kernel;
    int block_dim;
    int num_sms;
    int concurrency;
    int tag;
    float latency;
    std::vector<float> samples;
};

struct DiskCacheHeader
{
    char magic[8];
    uint32_t format;
    uint32_t record_size;
};

struct DiskCacheRecordV1
{
    uint64_t hash;
    int32_t tag;
    int32_t block_dim;
    int32_t num_sms;
    int32_t concurrency;
    uint32_t reserved0;
    uint32_t reserved1;
};

using LaunchKey = std::pair<int, uint64_t>;
std::map<LaunchKey, CoopAutotuneLaunch> launch_cache;
std::mutex launch_mutex;
std::mutex attr_mutex;

std::set<std::tuple<int, void*, size_t>> attr_set;

std::mutex disk_mutex;
std::map<uint64_t, DiskCacheRecordV1> disk_cache;

// EXL3_DEBUG_AUTOTUNE: print every probe/measurement launch and synchronize after it, so a launch that
// never completes (a grid-barrier kernel past true co-residency) is named instead of hanging silently
static bool debug_autotune()
{
    static const bool v = std::getenv("EXL3_DEBUG_AUTOTUNE") != nullptr;
    return v;
}

// Cache discovery per host thread and device. In particular, avoid reading
// the Linux driver build or querying the runtime during graph replay setup.
LaunchKey launch_key(uint64_t shape)
{
    int device;
    cuda_check(cudaGetDevice(&device));
    thread_local std::map<int, uint64_t> device_keys;
    auto it = device_keys.find(device);
    if (it == device_keys.end())
    {
        uint64_t h = COOP_AUTOTUNE_VERSION;
        auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
        const auto* prop = at::cuda::getDeviceProperties(device);
        for (const char* p = prop->name; *p; ++p) mix((unsigned char)*p);
        mix(prop->major);
        mix(prop->minor);
        int driver;
        cuda_check(cudaDriverGetVersion(&driver));
        mix(driver);
        #if defined(__linux__) && !defined(USE_ROCM)
        // The CUDA API compatibility version stays constant across driver patches.
        std::string build;
        std::ifstream version("/proc/driver/nvidia/version");
        if (std::getline(version, build))
            for (unsigned char ch : build) mix(ch);
        #endif
        it = device_keys.emplace(device, h).first;
    }
    return {device, (shape ^ it->second) * 1099511628211ull};
}

std::filesystem::path disk_cache_path()
{
    const char* override_path = std::getenv("EXLLAMAV3_TUNE_CACHE");
    if (override_path && override_path[0])
    {
        std::filesystem::path path = std::filesystem::path(override_path);
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec))
            return path / "coop_autotune_v1.bin";
        return path;
    }

    std::filesystem::path base;
    #ifdef _WIN32
    const char* local_app_data = std::getenv("LOCALAPPDATA");
    if (local_app_data && local_app_data[0])
        base = std::filesystem::path(local_app_data);
    else
    {
        const char* user_profile = std::getenv("USERPROFILE");
        if (!user_profile || !user_profile[0]) return {};
        std::filesystem::path user_path = std::filesystem::path(user_profile);
        std::filesystem::path local_path = user_path / "AppData" / "Local";
        std::error_code ec;
        if (std::filesystem::exists(local_path, ec))
            base = local_path;
        else
            base = user_path;
    }

    return base / "exllamav3" / "autotune" / "coop_autotune_v1.bin";
    #else
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0])
        base = std::filesystem::path(xdg);
    else
    {
        const char* home = std::getenv("HOME");
        if (!home || !home[0]) return {};
        base = std::filesystem::path(home) / ".cache";
    }

    return base / "exllamav3" / "autotune" / "coop_autotune_v1.bin";
    #endif
}

// Serialize a first tuning run with other processes sharing this file.
// The OS releases the lock on exit, including an interrupted calibration.
class PlanFileLock
{
    #ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    OVERLAPPED overlap{};
    #else
    int fd = -1;
    #endif
public:
    PlanFileLock(const PlanFileLock&) = delete;
    PlanFileLock& operator=(const PlanFileLock&) = delete;
    bool locked = false;
    explicit PlanFileLock(std::filesystem::path path)
    {
        if (path.empty()) return;
        std::error_code ec;
        if (!path.parent_path().empty())
            std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) return;
        path += ".lock";
        #ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, 0, nullptr);
        if (handle != INVALID_HANDLE_VALUE)
            locked = LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &overlap);
        #else
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (fd >= 0)
        {
            int rc;
            do { rc = flock(fd, LOCK_EX); } while (rc == -1 && errno == EINTR);
            locked = rc == 0;
        }
        #endif
    }
    ~PlanFileLock()
    {
        #ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
        #else
        if (fd >= 0) close(fd);
        #endif
    }
};

bool read_disk_header(std::ifstream& in)
{
    DiskCacheHeader header;
    in.read((char*) &header, sizeof(header));
    if (!in) return false;
    if (std::memcmp(header.magic, DISK_CACHE_MAGIC, sizeof(header.magic)) != 0) return false;
    if (header.format != DISK_CACHE_FORMAT) return false;
    if (header.record_size != sizeof(DiskCacheRecordV1)) return false;
    return true;
}

bool load_disk_cache_locked()
{
    disk_cache.clear();

    std::filesystem::path path = disk_cache_path();
    if (path.empty()) return false;

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::error_code ec;
        return !std::filesystem::exists(path, ec) && !ec;
    }
    if (in.peek() == std::ifstream::traits_type::eof()) return in.eof();
    // Preserve files with an unknown format, including newer cache revisions.
    if (!read_disk_header(in)) return false;

    DiskCacheRecordV1 record;
    while (in.read((char*) &record, sizeof(record)))
    {
        // Legacy append-only files could contain a second writer's header.
        if (std::memcmp(&record, DISK_CACHE_MAGIC, sizeof(DISK_CACHE_MAGIC)) == 0)
        {
            in.seekg(-(std::streamoff)(sizeof(record) - sizeof(DiskCacheHeader)), std::ios::cur);
            continue;
        }
        disk_cache[record.hash] = record;
    }

    #ifdef CACHEDEBUG
    printf
    (
        "coop_autotune cache loaded: path=%s records=%zu\n",
        path.string().c_str(),
        disk_cache.size()
    );
    #endif
    return true;
}

// Publish a complete cache. A stopped writer leaves the previous set of
// plans readable. The lock also protects the temporary name.
bool write_disk_cache_locked()
{
    auto path = disk_cache_path();
    auto tmp = path;
    tmp += ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    DiskCacheHeader header{};
    std::memcpy(header.magic, DISK_CACHE_MAGIC, sizeof(header.magic));
    header.format = DISK_CACHE_FORMAT;
    header.record_size = sizeof(DiskCacheRecordV1);
    out.write((const char*) &header, sizeof(header));
    for (const auto& entry : disk_cache)
        out.write((const char*) &entry.second, sizeof(entry.second));
    out.close();
    if (!out) return false;
    #ifdef _WIN32
    return MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
    #else
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
    #endif
}

bool launch_from_disk_cache
(
    uint64_t hash,
    const std::vector<CoopAutotuneCandidate>& candidates,
    CoopAutotuneLaunch* launch_config
)
{
    auto lookup = disk_cache.find(hash);
    if (lookup == disk_cache.end()) return false;

    const DiskCacheRecordV1& record = lookup->second;
    #ifdef CACHEDEBUG
    printf
    (
        "coop_autotune cache lookup: hash=%016llx tag=%d block_dim=%d num_sms=%d concurrency=%d\n",
        (unsigned long long) record.hash,
        record.tag,
        record.block_dim,
        record.num_sms,
        record.concurrency
    );
    #endif

    for (const CoopAutotuneCandidate& candidate : candidates)
    {
        if (candidate.tag != record.tag) continue;
        if (candidate.block_dim != record.block_dim) continue;
        if (record.num_sms < 1 || record.num_sms > candidate.max_num_sms) continue;

        int max_concurrency = MAX(candidate.max_concurrency, 1);
        int total_sms = candidate.total_sms > 0 ? candidate.total_sms : candidate.max_num_sms;
        int expected_concurrency = MAX(MIN(total_sms / record.num_sms, max_concurrency), 1);
        if (record.concurrency != expected_concurrency) continue;

        *launch_config =
        {
            candidate.kernel,
            record.block_dim,
            record.num_sms,
            record.concurrency,
            record.tag
        };
        #ifdef CACHEDEBUG
        printf
        (
            "coop_autotune cache hit: hash=%016llx tag=%d block_dim=%d num_sms=%d concurrency=%d\n",
            (unsigned long long) record.hash,
            launch_config->tag,
            launch_config->block_dim,
            launch_config->num_sms,
            launch_config->concurrency
        );
        #endif
        return true;
    }

    #ifdef CACHEDEBUG
    printf
    (
        "coop_autotune cache rejected: hash=%016llx tag=%d block_dim=%d num_sms=%d concurrency=%d\n",
        (unsigned long long) record.hash,
        record.tag,
        record.block_dim,
        record.num_sms,
        record.concurrency
    );
    #endif
    return false;
}

bool store_disk_cache(uint64_t hash, const CoopAutotuneLaunch& launch_config)
{
    DiskCacheRecordV1 record =
    {
        hash,
        launch_config.tag,
        launch_config.block_dim,
        launch_config.num_sms,
        launch_config.concurrency,
        0,
        0
    };

    disk_cache[hash] = record;

    #ifdef CACHEDEBUG
    printf
    (
        "coop_autotune cache store: hash=%016llx tag=%d block_dim=%d num_sms=%d concurrency=%d\n",
        (unsigned long long) record.hash,
        record.tag,
        record.block_dim,
        record.num_sms,
        record.concurrency
    );
    #endif
    return write_disk_cache_locked();
}

void set_kernel_attr_once(void* kernel, size_t smem)
{
    int device;
    cuda_check(cudaGetDevice(&device));

    std::lock_guard<std::mutex> lock(attr_mutex);
    auto key = std::make_tuple(device, kernel, smem);
    if (attr_set.find(key) != attr_set.end()) return;

    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
    attr_set.insert(key);
}

// Candidates in one autotune session share the same input tensors, so back-to-back timed launches
// keep each other's operands mutually warm in L2 regardless of production's actual scattered,
// per-token-random expert access pattern. The buffer lives only for the duration of one tuning
// session and is allocated through the Torch caching allocator: a raw cudaMalloc would fail
// whenever Torch has the device's VRAM reserved in its pool, even with plenty of it free.
struct ThrashBuffer
{
    at::Tensor tensor;
    void* ptr = nullptr;
    size_t size = 0;
};

ThrashBuffer alloc_thrash_buffer()
{
    int device;
    cuda_check(cudaGetDevice(&device));

    int l2_bytes = 0;
    cuda_check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
    size_t base = (size_t) (l2_bytes > 0 ? l2_bytes : 64 * 1024 * 1024);

    // 2x the reported L2 capacity to reliably evict prior residents regardless of
    // associativity/replacement-policy quirks; fall back to 1x, then to no thrashing at all,
    // if the device is too full (a warm-cache-biased tune beats an OOM crash)
    for (size_t size : { base * 2, base })
    {
        try
        {
            ThrashBuffer buf;
            buf.tensor = at::empty
            (
                { (int64_t) size },
                at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device)
            );
            buf.ptr = buf.tensor.data_ptr();
            buf.size = size;
            return buf;
        }
        catch (const c10::Error&) {}
    }
    TORCH_WARN
    (
        "CoopKernelAutotuner: could not allocate an L2 thrash buffer on device ", device,
        "; timing candidates with warm cache"
    );
    return {};
}

void thrash_l2(const ThrashBuffer& buf, cudaStream_t stream)
{
    if (!buf.ptr) return;
    cuda_check(cudaMemsetAsync(buf.ptr, 0, buf.size, stream));
}

float trimmed_mean(std::vector<float>& samples)
{
    TORCH_CHECK(!samples.empty(), "CoopKernelAutotuner: no timing samples");
    std::sort(samples.begin(), samples.end());

    size_t trim = samples.size() / 4;
    size_t begin = trim;
    size_t end = samples.size() - trim;
    if (begin >= end)
    {
        begin = 0;
        end = samples.size();
    }

    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += samples[i];
    return (float) (sum / (double) (end - begin));
}

void measure_candidate_sample
(
    ExpandedCandidate& candidate,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    int repeats,
    cudaEvent_t start,
    cudaEvent_t end,
    const ThrashBuffer& thrash_buf
)
{
    thrash_l2(thrash_buf, stream);
    cuda_check(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i)
    {
        cuda_check(EXL3_COOP_LAUNCH
        (
            candidate.kernel,
            dim3(candidate.num_sms, 1, candidate.concurrency),
            candidate.block_dim,
            kernel_args,
            smem,
            stream
        ));
        if (debug_autotune())
        {
            fprintf(stderr, "[measure] grid=%dx%d block=%d rep=%d/%d ... ",
                    candidate.num_sms, candidate.concurrency, candidate.block_dim, i + 1, repeats);
            fflush(stderr);
            cuda_check(cudaStreamSynchronize(stream));
            fprintf(stderr, "ok\n");
        }
    }
    cuda_check(cudaEventRecord(end, stream));
    cuda_check(cudaEventSynchronize(end));

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, end));
    candidate.samples.push_back(ms / (float) repeats);
}

void keep_best(std::vector<ExpandedCandidate>& candidates, size_t limit)
{
    for (ExpandedCandidate& candidate : candidates)
        candidate.latency = trimmed_mean(candidate.samples);

    std::sort
    (
        candidates.begin(),
        candidates.end(),
        [] (const ExpandedCandidate& a, const ExpandedCandidate& b)
        {
            return a.latency < b.latency;
        }
    );
    if (candidates.size() > limit) candidates.resize(limit);
}

void measure_stage
(
    std::vector<ExpandedCandidate>& candidates,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    int rounds,
    int repeats,
    size_t keep,
    cudaEvent_t start,
    cudaEvent_t end,
    const ThrashBuffer& thrash_buf
)
{
    TORCH_CHECK(!candidates.empty(), "CoopKernelAutotuner: no candidates in stage");

    for (ExpandedCandidate& candidate : candidates)
    {
        candidate.samples.clear();
        candidate.samples.reserve(rounds);
        set_kernel_attr_once(candidate.kernel, smem);

        // One untimed launch avoids first-use effects from contaminating the first measured round.
        cuda_check(EXL3_COOP_LAUNCH
        (
            candidate.kernel,
            dim3(candidate.num_sms, 1, candidate.concurrency),
            candidate.block_dim,
            kernel_args,
            smem,
            stream
        ));
        if (debug_autotune())
        {
            fprintf(stderr, "[autotune probe] kernel=%p grid=%dx%dx%d block=%d smem=%zu ... ",
                    candidate.kernel, candidate.num_sms, 1, candidate.concurrency, candidate.block_dim, smem);
            fflush(stderr);
            cuda_check(cudaStreamSynchronize(stream));
            fprintf(stderr, "ok\n");
        }
    }
    cuda_check(cudaStreamSynchronize(stream));

    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);

    for (int round = 0; round < rounds; ++round)
    {
        size_t n = order.size();
        size_t step = 2 * (size_t) round + 1;
        while (std::gcd(step, n) != 1) step += 2;
        size_t pos = ((uint64_t) round * 1103515245ull + 12345ull) % n;

        for (size_t i = 0; i < n; ++i)
        {
            measure_candidate_sample
            (
                candidates[order[pos]],
                kernel_args,
                smem,
                stream,
                repeats,
                start,
                end,
                thrash_buf
            );
            pos = (pos + step) % n;
        }
    }

    keep_best(candidates, keep);
}

CoopAutotuneLaunch tune
(
    const std::vector<CoopAutotuneCandidate>& base_candidates,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    size_t numel_B
)
{
    std::vector<ExpandedCandidate> candidates;
    for (const CoopAutotuneCandidate& base : base_candidates)
    {
        int total_tiles = numel_B / base.block_dim / 16;
        int min_num_sms = MAX(2, MIN(total_tiles / 32, base.max_num_sms) / base.max_concurrency / 2 * 2);

        TORCH_CHECK(base.kernel, "CoopKernelAutotuner: null kernel candidate");
        TORCH_CHECK(base.block_dim > 0, "CoopKernelAutotuner: invalid block_dim");
        TORCH_CHECK(base.max_num_sms > 0, "CoopKernelAutotuner: invalid max_num_sms");
        int max_concurrency = MAX(base.max_concurrency, 1);
        int total_sms = base.total_sms > 0 ? base.total_sms : base.max_num_sms;

        if (max_concurrency > 1 || base.max_num_sms == 1)
        {
            int concurrency = MAX(MIN(total_sms, max_concurrency), 1);
            candidates.push_back({ base.kernel, base.block_dim, 1, concurrency, base.tag, 0.0f, {} });
        }

        for (int num_sms = min_num_sms; num_sms <= base.max_num_sms * 85 / 100; num_sms += 2)
        {
            int concurrency = MAX(MIN(total_sms / num_sms, max_concurrency), 1);
            candidates.push_back({ base.kernel, base.block_dim, num_sms, concurrency, base.tag, 0.0f, {} });
        }

        if (base.max_num_sms > 1)
        {
            int concurrency = MAX(MIN(total_sms / base.max_num_sms, max_concurrency), 1);
            candidates.push_back({ base.kernel, base.block_dim, base.max_num_sms, concurrency, base.tag, 0.0f, {} });
        }
    }
    TORCH_CHECK(!candidates.empty(), "CoopKernelAutotuner: no candidates");

    // One buffer for the whole session, freed (back to the Torch pool) when tune() returns
    ThrashBuffer thrash_buf = alloc_thrash_buffer();

    cudaEvent_t start;
    cudaEvent_t end;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&end));

    int repeats = 8;
    if (numel_B > 1e6) repeats = 5;
    if (numel_B > 1e7) repeats = 4;
    if (numel_B > 1e8) repeats = 3;
    if (numel_B > 2e8) repeats = 2;
    int max_rounds = 32;
    if (numel_B > 1e6) max_rounds = 12;
    if (numel_B > 1e7) max_rounds = 8;
    if (numel_B > 1e8) max_rounds = 5;
    if (numel_B > 2e8) max_rounds = 2;
    int max_cands = 7;
    if (numel_B > 1e6) max_cands = 5;
    if (numel_B > 1e7) max_cands = 4;
    if (numel_B > 1e8) max_cands = 3;
    if (numel_B > 2e8) max_cands = 2;

    if (candidates.size() > 1 && max_cands > 4)
        measure_stage(candidates, kernel_args, smem, stream, MIN(2, max_rounds), repeats, MIN(6, max_cands), start, end, thrash_buf);
    if (candidates.size() > 1 && max_cands > 1)
        measure_stage(candidates, kernel_args, smem, stream, MIN(8, max_rounds), repeats, MIN(3, max_cands), start, end, thrash_buf);
    if (candidates.size() > 1)
        measure_stage(candidates, kernel_args, smem, stream, MIN(14, max_rounds), repeats, 1, start, end, thrash_buf);

    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(end));

    const ExpandedCandidate& best = candidates[0];
    return { best.kernel, best.block_dim, best.num_sms, best.concurrency, best.tag };
}

}  // namespace

bool CoopKernelAutotuner::launch_locked
(
    uint64_t hash,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    CoopAutotuneLaunch* out_launch_config
)
{
    CoopAutotuneLaunch launch_config;
    auto key = launch_key(hash);

    {
        std::lock_guard<std::mutex> lock(launch_mutex);
        auto lookup = launch_cache.find(key);
        if (lookup == launch_cache.end()) return false;
        launch_config = lookup->second;
    }

    set_kernel_attr_once(launch_config.kernel, smem);
    cuda_check(EXL3_COOP_LAUNCH
    (
        launch_config.kernel,
        dim3(launch_config.num_sms, 1, launch_config.concurrency),
        launch_config.block_dim,
        kernel_args,
        smem,
        stream
    ));

    if (out_launch_config) *out_launch_config = launch_config;
    return true;
}

CoopAutotuneLaunch CoopKernelAutotuner::launch
(
    uint64_t hash,
    const std::vector<CoopAutotuneCandidate>& candidates,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    size_t numel_B
)
{
    CoopAutotuneLaunch launch_config;
    auto key = launch_key(hash);
    {
        std::lock_guard<std::mutex> cold_lock(disk_mutex);
        if (launch_locked(hash, kernel_args, smem, stream, &launch_config))
            return launch_config;

        PlanFileLock file_lock(disk_cache_path());
        bool usable_file = load_disk_cache_locked();
        bool hit = launch_from_disk_cache(key.second, candidates, &launch_config);
        if (!hit)
        {
            launch_config = tune(candidates, kernel_args, smem, stream, numel_B);
            if (!usable_file || !file_lock.locked || !store_disk_cache(key.second, launch_config))
                TORCH_WARN_ONCE("Could not persist cooperative GEMM tuning at ", disk_cache_path().string());
        }
        if (debug_autotune())
            fprintf(stderr, "[autotune %s] key=%016llx shape=%d sms=%d concurrent=%d\n",
                    hit ? "hit" : "miss", (unsigned long long) key.second, launch_config.tag,
                    launch_config.num_sms, launch_config.concurrency);
        std::lock_guard<std::mutex> lock(launch_mutex);
        launch_cache.emplace(key, launch_config);
    }

    set_kernel_attr_once(launch_config.kernel, smem);
    cuda_check(EXL3_COOP_LAUNCH
    (
        launch_config.kernel,
        dim3(launch_config.num_sms, 1, launch_config.concurrency),
        launch_config.block_dim,
        kernel_args,
        smem,
        stream
    ));

    return launch_config;
}
