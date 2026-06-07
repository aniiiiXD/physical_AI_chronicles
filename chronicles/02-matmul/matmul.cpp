#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

// ─── matrix stored in row-major order ────────────────────────────────────────
// Element (i,j) of an N×N matrix lives at data[i*N + j].
// Row-major means a whole row is contiguous in memory — ideal for C.

// ─── naïve triple-loop matmul ─────────────────────────────────────────────────
// The classic O(N³) algorithm written straight from the definition:
//   C[i][j] = sum_k  A[i][k] * B[k][j]
//
// The problem: B is accessed column-by-column (stride = N floats = N*4 bytes).
// For large N that stride is larger than a cache line, so each B[k][j] access
// is a cache miss — we pay RAM latency N³ times.
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

// ─── cache-blocked (tiled) matmul ────────────────────────────────────────────
// Idea: instead of iterating over the full i/j/k ranges, break each dimension
// into tiles of size BLOCK.  Within a tile, all data fits in L1/L2 cache,
// so we reuse it many times before it gets evicted.
//
// Outer loops: step through tiles.
// Inner loops: work inside one tile — same triple loop, just over BLOCK×BLOCK.
//
// With blocking:
//   - A tile of A  is BLOCK×BLOCK floats = BLOCK² * 4 bytes
//   - A tile of B  is BLOCK×BLOCK floats
//   - A tile of C  is BLOCK×BLOCK floats
//   Total working set ≈ 3 * BLOCK² * 4 bytes
//   For BLOCK=64: 3 * 64² * 4 = 49 152 bytes ≈ 48 KB — fits in L2 (~256 KB)
//   For BLOCK=32: 3 * 32² * 4 = 12 288 bytes ≈ 12 KB — fits in L1 (~32 KB)
//
// We accumulate into C rather than zeroing inside the tile loop because tiles
// of A and B together contribute partial sums to the same tile of C.
void matmul_blocked(const float* A, const float* B, float* C, int N, int BLOCK) {
    // Zero C first — we accumulate partial sums across K-tiles.
    std::memset(C, 0, N * N * sizeof(float));

    for (int i0 = 0; i0 < N; i0 += BLOCK)          // tile row of A / C
    for (int j0 = 0; j0 < N; j0 += BLOCK)          // tile col of B / C
    for (int k0 = 0; k0 < N; k0 += BLOCK) {        // tile col of A / row of B

        // Clamp tile edges so we don't go out of bounds when N % BLOCK != 0.
        int i_end = std::min(i0 + BLOCK, N);
        int j_end = std::min(j0 + BLOCK, N);
        int k_end = std::min(k0 + BLOCK, N);

        // ── inner micro-kernel: works on one BLOCK×BLOCK tile ──────────────
        for (int i = i0; i < i_end; ++i)
        for (int k = k0; k < k_end; ++k) {
            float a = A[i*N + k];                   // load once, reuse for all j
            for (int j = j0; j < j_end; ++j)
                C[i*N + j] += a * B[k*N + j];      // B row is contiguous — good
        }
        // Note the i/k/j order (not i/j/k): hoisting A[i*N+k] out of the j
        // loop means one A load serves BLOCK multiply-adds — this is the
        // "register reuse" trick.  B[k*N + j] is now accessed row-by-row
        // (stride 1), which is cache-friendly.
    }
}

// ─── timing helper ───────────────────────────────────────────────────────────
static double now_ns() {
    return std::chrono::duration<double, std::nano>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// ─── correctness check ───────────────────────────────────────────────────────
static bool matrices_close(const float* A, const float* B, int N, float tol = 1e-3f) {
    for (int i = 0; i < N * N; ++i)
        if (std::fabs(A[i] - B[i]) > tol * std::max(1.0f, std::fabs(A[i])))
            return false;
    return true;
}

// ─── GFLOP/s calculator ──────────────────────────────────────────────────────
// Matrix multiply of N×N matrices does 2*N³ floating-point operations
// (N³ multiplications + N³ additions). We divide by elapsed seconds and 1e9
// to get giga-FLOP/s — the standard performance metric for dense linear algebra.
static double gflops(int N, double seconds) {
    return 2.0 * N * N * N / seconds / 1e9;
}

int main() {
    // Matrix sizes to benchmark — skipping very large N for blocked because
    // it would take too long at this demo level without SIMD/threading.
    const int sizes[]  = { 64, 128, 256, 512, 1024 };
    const int blocks[] = { 32, 64 };          // tile sizes to try

    // Header
    std::cout << std::left
              << std::setw(6)  << "N"
              << std::setw(12) << "Naive(ms)"
              << std::setw(14) << "Naive(GF/s)";
    for (int b : blocks)
        std::cout << std::setw(14) << ("B"+std::to_string(b)+"(ms)")
                  << std::setw(14) << ("B"+std::to_string(b)+"(GF/s)");
    std::cout << "Speedup(B64)\n" << std::string(90, '-') << '\n';

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int N : sizes) {
        int sz = N * N;
        std::vector<float> A(sz), B(sz), C_naive(sz, 0), C_blocked(sz, 0);

        // Fill A and B with random floats.
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        // ── naïve ────────────────────────────────────────────────────────────
        // Run multiple times for small N to get a stable measurement.
        int reps = std::max(1, 512 / (N + 1));
        double t0 = now_ns();
        for (int r = 0; r < reps; ++r) matmul_naive(A.data(), B.data(), C_naive.data(), N);
        double naive_ms = (now_ns() - t0) / 1e6 / reps;

        std::cout << std::left << std::fixed << std::setprecision(2)
                  << std::setw(6)  << N
                  << std::setw(12) << naive_ms
                  << std::setw(14) << gflops(N, naive_ms * 1e-3);

        double last_blocked_ms = naive_ms;

        for (int BLOCK : blocks) {
            // ── blocked ───────────────────────────────────────────────────────
            double tb0 = now_ns();
            for (int r = 0; r < reps; ++r) matmul_blocked(A.data(), B.data(), C_blocked.data(), N, BLOCK);
            double blocked_ms = (now_ns() - tb0) / 1e6 / reps;
            last_blocked_ms = blocked_ms;

            // Verify only on first block size (same A,B so result is identical)
            bool ok = matrices_close(C_naive.data(), C_blocked.data(), N);

            std::cout << std::setw(14) << blocked_ms
                      << std::setw(13) << gflops(N, blocked_ms * 1e-3)
                      << (ok ? " " : " MISMATCH ");
        }

        std::cout << std::setw(6) << (naive_ms / last_blocked_ms) << "x\n";
        std::cout.flush();
    }
}
