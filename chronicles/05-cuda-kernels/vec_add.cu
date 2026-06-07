/*
 * vec_add.cu — vector addition on RTX 3060 (sm_86)
 *
 * Measures:
 *   - correctness vs CPU reference
 *   - achieved memory bandwidth  (target: ~330 GB/s theoretical peak ~360 GB/s)
 *
 * Build:
 *   cmake --build build --target vec_add
 * Run:
 *   ./build/vec_add
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ── error checking ────────────────────────────────────────────────────────────
#define CK(call) do {                                                          \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(e));                    \
        exit(1);                                                               \
    }                                                                          \
} while (0)

// ── kernel ────────────────────────────────────────────────────────────────────
// Each thread handles one element. Memory access is perfectly coalesced:
// thread 0,1,2,...,255 read consecutive addresses → single 1024-byte transaction.
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float*       __restrict__ c,
                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ── CUDA event timer ──────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start()               { cudaEventRecord(s); }
    float stop_ms()            { cudaEventRecord(e); cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e); return ms; }
};

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    const int  N        = 1 << 26;   // 64 M floats = 256 MB per array
    const int  REPS     = 20;
    const int  THREADS  = 256;
    const int  BLOCKS   = (N + THREADS - 1) / THREADS;

    // host arrays
    std::vector<float> h_a(N), h_b(N), h_c(N), h_ref(N);
    for (int i = 0; i < N; i++) { h_a[i] = (float)i * 0.5f; h_b[i] = (float)i * 1.5f; }
    for (int i = 0; i < N; i++) h_ref[i] = h_a[i] + h_b[i];

    // device arrays
    float *d_a, *d_b, *d_c;
    CK(cudaMalloc(&d_a, N * sizeof(float)));
    CK(cudaMalloc(&d_b, N * sizeof(float)));
    CK(cudaMalloc(&d_c, N * sizeof(float)));
    CK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // warmup
    vec_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
    CK(cudaDeviceSynchronize());

    // benchmark
    GpuTimer timer;
    float total_ms = 0.0f;
    for (int r = 0; r < REPS; r++) {
        timer.start();
        vec_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
        total_ms += timer.stop_ms();
    }
    float avg_ms = total_ms / REPS;

    // bandwidth:  3 arrays read+write, each N floats of 4 bytes
    double bytes    = 3.0 * N * sizeof(float);
    double bw_gbs   = bytes / (avg_ms * 1e-3) / 1e9;

    // correctness check
    CK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) max_err = fmaxf(max_err, fabsf(h_c[i] - h_ref[i]));

    printf("\n=== vec_add — RTX 3060 ===\n");
    printf("  N              : %d (%.0f MB per array)\n", N, N * 4.0 / 1e6);
    printf("  threads/block  : %d\n", THREADS);
    printf("  blocks         : %d\n", BLOCKS);
    printf("  avg latency    : %.3f ms\n", avg_ms);
    printf("  bandwidth      : %.1f GB/s  (peak ~360 GB/s)\n", bw_gbs);
    printf("  max |error|    : %.2e  %s\n\n", max_err, max_err < 1e-5f ? "✓ PASS" : "✗ FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
