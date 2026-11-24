#pragma once

// A GB10 arithmetic plan describes how to group products. The autotuner
// independently chooses how many physical CTAs execute those groups.
struct GB10ReductionPlan
{
    int tile_k;
    int tile_n;
    int slices;
};

__host__ __device__ inline GB10ReductionPlan gb10_reduction_plan(
    int rows, int k, int n, int bits, bool half_k, bool c_fp32, int cb, int matrices)
{
    int tile_k = k % 32 == 0 ? 32 : 16;
    int tile_n = 128;
    int wave = 48 / MIN(matrices, 48);
    int slices = MIN(wave, MIN(4, k / tile_k) * (n / tile_n));

    // The qualified 6-bit Inkling verifier uses different reduction trees
    // from prefill. Preserve its already-rounded results at widths 1..8.
    if (rows <= 8 && bits == 6 && !half_k && !c_fp32 && cb == 2)
    {
        if (matrices == 1 && k == 2304 && n == 3072) return {32, 256, 48};
        if (matrices == 1 && k == 2304 && n == 1024) return {32, 128, 30};
        if (matrices == 1 && k == 3072 && n == 2304) return {32, 256, 40};
        if (matrices == 2 && k == 2304 && n == 3072) return {16, 512, 24};
        if (matrices == 2 && k == 2304 && n == 1024) return {32, 128, 24};
    }
    return {tile_k, tile_n, slices};
}
