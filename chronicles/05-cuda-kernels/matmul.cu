/*
 * matmul.cu — three matmul implementations on RTX 3060 (sm_86)
 *
 *   1. naive   — one thread per output element, no shared memory
 *   2. tiled   — 32×32 shared-memory tiles (TILE=32)
 *   3. cuBLAS  — library baseline (uses tensor cores internally)
 *
 * Theory recap (from 04-memory-deep-dive):
 *   Naive:  every element of A and B is loaded N times from global mem.
 *           Memory traffic = 2 * N^3 * 4 bytes → memory-bound.
 *   Tiled:  load a TILE×TILE block of A and B into __shared__ once,
 *           reuse it TILE times. Reduces global mem traffic by TILE×.
 *           With TILE=32 each SM loads 32 KB of shared mem per step.
 *
 * Build:
 *   cmake --build build --target matmul
 * Run:
 *   ./build/matmul
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(e)); exit(1);                               \
    }                                                                          \
} while (0)

#define CUBLAS_CK(call) do {                                                   \
    cublasStatus_t s = (call);                                                 \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s);   \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// ── 1. naive kernel ───────────────────────────────────────────────────────────
// Thread (row, col) accumulates the full dot product A[row,:] · B[:,col].
// Every A[row,k] is re-loaded N times — one load per output in the same row.
__global__ void matmul_naive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ── 2. tiled kernel ───────────────────────────────────────────────────────────
// Shared memory layout:
//   As[TILE][TILE] — current A tile (row-major slice of A)
//   Bs[TILE][TILE] — current B tile (row-major slice of B)
//
// Each thread loads exactly one element into As and one into Bs.
// After __syncthreads(), all 32×32 threads compute their partial dot product
// using the tile data that is now in the fast shared memory (~5 cycles).
// Then we move to the next tile. Total tiles per row = N / TILE.
//
// Shared mem per block: 2 * 32 * 32 * 4 = 8 KB.
// RTX 3060 shared mem per SM: up to 100 KB → can hold ~12 blocks simultaneously.

constexpr int TILE = 32;

__global__ void matmul_tiled(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // 4 independent accumulators break the serial FMA dependency chain.
    // Ampere FMA latency = 4 cycles. With one acc, each iteration stalls
    // waiting for the previous result. With 4 independent accs the GPU
    // dispatches all four FMAs simultaneously, hiding that latency.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k += 4) {
            acc0 += As[threadIdx.y][k+0] * Bs[k+0][threadIdx.x];
            acc1 += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
            acc2 += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
            acc3 += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = acc0 + acc1 + acc2 + acc3;
}

// ── timer ─────────────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void  start()    { cudaEventRecord(s); }
    float stop_ms()  { cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e); return ms; }
};

// ── helpers ───────────────────────────────────────────────────────────────────
static double gflops(int N, float ms) {
    return 2.0 * N * N * N / (ms * 1e-3) / 1e9;
}

static float max_rel_err(const std::vector<float>& ref, const std::vector<float>& got) {
    float err = 0.0f;
    for (size_t i = 0; i < ref.size(); i++) {
        float e = fabsf(ref[i] - got[i]) / (fabsf(ref[i]) + 1e-6f);
        err = fmaxf(err, e);
    }
    return err;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    const int N    = 4096;   // 4096×4096 — enough to saturate the 3060
    const int REPS = 10;
    const long SZ  = (long)N * N;

    printf("\n=== matmul — RTX 3060  N=%d ===\n\n", N);

    // host data
    std::vector<float> h_A(SZ), h_B(SZ), h_C_naive(SZ), h_C_tiled(SZ), h_C_cublas(SZ);
    for (int i = 0; i < SZ; i++) { h_A[i] = (float)rand() / RAND_MAX; h_B[i] = (float)rand() / RAND_MAX; }

    // device alloc
    float *d_A, *d_B, *d_naive, *d_tiled, *d_cublas;
    CK(cudaMalloc(&d_A,      SZ * sizeof(float)));
    CK(cudaMalloc(&d_B,      SZ * sizeof(float)));
    CK(cudaMalloc(&d_naive,  SZ * sizeof(float)));
    CK(cudaMalloc(&d_tiled,  SZ * sizeof(float)));
    CK(cudaMalloc(&d_cublas, SZ * sizeof(float)));
    CK(cudaMemcpy(d_A, h_A.data(), SZ * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_B, h_B.data(), SZ * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // ── 1. naive ────────────────────────────────────────────────────────────
    matmul_naive<<<grid, block>>>(d_A, d_B, d_naive, N);  // warmup
    CK(cudaDeviceSynchronize());

    float t_naive = 0;
    for (int r = 0; r < REPS; r++) {
        timer.start();
        matmul_naive<<<grid, block>>>(d_A, d_B, d_naive, N);
        t_naive += timer.stop_ms();
    }
    t_naive /= REPS;
    CK(cudaMemcpy(h_C_naive.data(), d_naive, SZ * sizeof(float), cudaMemcpyDeviceToHost));

    // ── 2. tiled ────────────────────────────────────────────────────────────
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_tiled, N);  // warmup
    CK(cudaDeviceSynchronize());

    float t_tiled = 0;
    for (int r = 0; r < REPS; r++) {
        timer.start();
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_tiled, N);
        t_tiled += timer.stop_ms();
    }
    t_tiled /= REPS;
    CK(cudaMemcpy(h_C_tiled.data(), d_tiled, SZ * sizeof(float), cudaMemcpyDeviceToHost));

    // ── 3. cuBLAS ───────────────────────────────────────────────────────────
    // cuBLAS is column-major. To compute C = A*B in row-major:
    // treat as C^T = B^T * A^T in col-major → just swap A,B in the call.
    cublasHandle_t handle;
    CUBLAS_CK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_B, N, d_A, N, &beta, d_cublas, N);  // warmup
    CK(cudaDeviceSynchronize());

    float t_cublas = 0;
    for (int r = 0; r < REPS; r++) {
        timer.start();
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_cublas, N);
        t_cublas += timer.stop_ms();
    }
    t_cublas /= REPS;
    cublasDestroy(handle);
    CK(cudaMemcpy(h_C_cublas.data(), d_cublas, SZ * sizeof(float), cudaMemcpyDeviceToHost));

    // ── correctness: compare tiled vs cuBLAS ────────────────────────────────
    float err_naive = max_rel_err(h_C_cublas, h_C_naive);
    float err_tiled = max_rel_err(h_C_cublas, h_C_tiled);

    // ── results ──────────────────────────────────────────────────────────────
    printf("  %-10s  %7.1f ms  %7.2f TFLOP/s  err=%.2e  %s\n",
           "naive",  t_naive,  gflops(N, t_naive)  / 1000, err_naive,
           err_naive < 1e-3f ? "✓" : "✗");
    printf("  %-10s  %7.1f ms  %7.2f TFLOP/s  err=%.2e  %s\n",
           "tiled",  t_tiled,  gflops(N, t_tiled)  / 1000, err_tiled,
           err_tiled < 1e-3f ? "✓" : "✗");
    printf("  %-10s  %7.1f ms  %7.2f TFLOP/s  (baseline)\n",
           "cuBLAS", t_cublas, gflops(N, t_cublas) / 1000);

    printf("\n  tiled vs naive  : %.1f×\n", t_naive / t_tiled);
    printf("  cuBLAS vs tiled : %.1f×\n\n", t_tiled / t_cublas);

    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_naive); cudaFree(d_tiled); cudaFree(d_cublas);
    return 0;
}
