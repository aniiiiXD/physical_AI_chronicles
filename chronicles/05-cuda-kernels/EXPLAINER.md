# CUDA Kernels — Line by Line Explainer

This chapter has five files: `vec_add.cu`, `matmul.cu`, `bench.py`,
`profile.sh`, and `CMakeLists.txt`. Together they cover the complete
"write → profile → compare to a library" loop for GPU programming.

---

## Big picture: what these programs are doing

```
vec_add.cu   → answers "how fast is raw memory?"         (bandwidth-bound)
matmul.cu    → answers "how fast is compute?"             (compute-bound)
bench.py     → answers "how much do Tensor Cores help?"   (fp16 vs fp32)
profile.sh   → answers "what is the hardware actually doing?" (hardware counters)
```

Each file builds on the last. vec_add establishes the memory bandwidth
ceiling. matmul shows how far you can get by reusing data (tiling) and
how far cuBLAS goes beyond that (tensor cores). bench.py adds the Python/
Triton angle. profile.sh gives you the hardware evidence to explain the gap.

---

## `vec_add.cu`

### The `CK` macro (lines 21–28)

```cpp
#define CK(call) do {
    cudaError_t e = (call);
    if (e != cudaSuccess) { fprintf(stderr, ...); exit(1); }
} while (0)
```

Every CUDA runtime call returns an error code. If you ignore it, a failed
`cudaMalloc` silently gives you a null pointer and your kernel crashes later
with a confusing segfault. `CK` wraps any call — `CK(cudaMalloc(...))` —
and aborts immediately with the exact file, line, and error string.

The `do { } while(0)` is a classic C macro pattern. It makes the macro
behave as a single statement, so `if (x) CK(foo); else bar();` works
correctly (without it, the `if` would only guard the `cudaError_t e = (call)`
line, not the rest of the macro body).

### The kernel (lines 33–40)

```cpp
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float*       __restrict__ c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**`__global__`**: this function runs on the GPU and is called from the CPU.
(The other qualifiers: `__device__` = GPU-only, `__host__` = CPU-only.)

**`__restrict__`**: tells the compiler that `a`, `b`, `c` never alias each
other (no pointer points into another's memory). This lets the compiler
generate better load/store schedules because it knows a write to `c` can't
change what's in `a` or `b`.

**Thread ID formula**:
```
i = blockIdx.x * blockDim.x + threadIdx.x
```
- `blockDim.x` = threads per block (256 in this program)
- `blockIdx.x` = which block is this? (0, 1, 2, …, BLOCKS-1)
- `threadIdx.x` = which thread within the block? (0–255)

Block 0's threads get i = 0–255. Block 1's threads get i = 256–511. The
formula tiles the thread indices across the entire array.

**`if (i < n)`**: guard for the last block. If N = 67,108,864 and we launch
exactly `ceil(N/256) * 256` threads, the last block's extra threads have
`i >= N` and must not write out of bounds.

### Why `3 × N × sizeof(float)` for bandwidth (line 87)

```cpp
double bytes = 3.0 * N * sizeof(float);
```

The operation `c[i] = a[i] + b[i]` does:
- Read `a[i]`: 1 load
- Read `b[i]`: 1 load
- Write `c[i]`: 1 store

Three memory transactions per element. The bandwidth formula is:
```
BW = total_bytes_moved / time = (3 × N × 4) / (ms × 1e-3) / 1e9
```
This gives GB/s, which you compare against the RTX 3060 Mobile's theoretical
peak of **192 GB/s**. Achieving 181 GB/s (94%) means the kernel is nearly
optimal — essentially no wasted bandwidth.

### `GpuTimer` (lines 43–49)

```cpp
struct GpuTimer {
    cudaEvent_t s, e;
    void start()       { cudaEventRecord(s); }
    float stop_ms()    { cudaEventRecord(e); cudaEventSynchronize(e);
                         float ms; cudaEventElapsedTime(&ms,s,e); return ms; }
};
```

`cudaEvent_t` is a timestamp inserted into the GPU's command queue. When
the GPU reaches that point in its execution stream, it records the time.
`cudaEventElapsedTime` returns the GPU-measured gap between two events.

**Why not `time.time()` or `std::chrono`?** Those measure wall-clock time on
the CPU. The GPU runs asynchronously — `cudaMemcpy` or a kernel launch
returns to the CPU immediately while the GPU is still working. Wall-clock
time includes CPU scheduling noise and async gaps. CUDA events are inserted
directly into the GPU command stream and measure only GPU execution time.

---

## `matmul.cu`

### The naive kernel (lines 47–60)

```cpp
__global__ void matmul_naive(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

Thread (row, col) computes the dot product of row `row` of A with column
`col` of B. The 2D indexing uses `blockIdx.y/x` and `threadIdx.y/x`.

**The fundamental problem**: for a 4096×4096 matrix, row 0 of A is read
once by thread (0,0), once by thread (0,1), once by thread (0,2)... a
total of 4096 times. Every element of A is loaded from DRAM **N times**.
Total memory traffic = 2 × N³ × 4 bytes ≈ 549 GB for N=4096.

The GPU's DRAM can move ~192 GB/s. At that rate this is a 2.8-second
operation. The actual time is ~300ms because the L2 cache catches some
of those redundant loads. But we're still massively memory-bottlenecked.

### The tiled kernel (lines 62–109)

The fix is **data reuse via shared memory**:

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
```

`__shared__` allocates memory in the SM's scratchpad (not DRAM). It is
shared by all threads in the same block and has ~20-cycle latency vs
~600-cycle DRAM latency.

**The tiling strategy**: instead of one thread computing an entire dot
product, the block of `TILE×TILE` threads cooperates. For each tile position
`t` along the K dimension:

1. Each thread loads one element of A and one element of B into shared memory
2. `__syncthreads()` — wait until everyone has loaded their element
3. All threads compute their partial dot product using the tile in shared memory
4. `__syncthreads()` — wait before overwriting the tile

After iterating over all tiles, each thread has its complete dot product.

**Why TILE=16?** With TILE=16, each block has 16×16 = 256 threads = 8 warps.
The SM can hold max 48 warps and max 32 blocks, so 6 blocks fit per SM
(6 × 8 = 48 warps = 100% occupancy). With TILE=32: 1024 threads, only 2
blocks per SM = 64 warps but limited by the block count limit to 2 × 32 =
64 warps... actually the 32-thread block limit (max 32 blocks/SM) limits you
to 32/32 × 1024 = wait, this gets complicated. The key point: TILE=16 gives
more warps in flight, which hides latency better.

### `__ldg()` (lines 96–99)

```cpp
As[threadIdx.y][threadIdx.x] = (row < N && aCol < N)
    ? __ldg(&A[row * N + aCol]) : 0.0f;
```

`__ldg` = load through the **read-only data cache** (the L1 texture cache).
On Ampere (sm_86), this is a separate 48 KB cache per SM that bypasses the
regular L1 coherence logic. Since A and B are never written by this kernel,
their data is always valid in this cache. Benefit: slightly higher cache hit
rate for the tile loads.

### cuBLAS and the column-major trick (lines 189–204)

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, d_B, N, d_A, N, &beta, d_cublas, N);
```

**cuBLAS assumes column-major storage**. C arrays are row-major (row 0
occupies addresses 0 to N-1, then row 1, etc.). Column-major is the
opposite (column 0 first).

The mathematical trick: if A is row-major, then the column-major view of A
is exactly A^T (the transpose). So:

```
C = A × B  in row-major
```
is the same computation as:
```
C^T = B^T × A^T  in column-major
```

And since cuBLAS's column-major view of our row-major B is B^T, passing
`d_B` first and `d_A` second gives us `C = A × B` in row-major. The output
`d_cublas` in row-major equals the correct answer.

**Why is cuBLAS 6–7× faster than our tiled kernel?**
1. **Tensor Cores**: cuBLAS uses WMMA instructions that compute a 16×16×16
   matrix multiply in a single operation, executing in ~8 cycles vs 256
   sequential FMA instructions.
2. **Double-buffering**: while computing tile T, it asynchronously loads
   tile T+1 using `cp.async` instructions that DMA directly to shared memory
   without stalling the SM.
3. **Register tiling**: inner loops are fully unrolled into registers,
   avoiding shared memory bank conflicts entirely.

### `gflops` helper (lines 121–123)

```cpp
static double gflops(int N, float ms) {
    return 2.0 * N * N * N / (ms * 1e-3) / 1e9;
}
```

An N×N matmul does N³ multiply-accumulate operations. Each MAC counts as
2 FLOPs (one multiply + one add), hence `2.0 * N³`. Divide by seconds and
by 1e9 to get GFLOP/s. Divide that by 1000 in the print statement for TFLOP/s.

### `max_rel_err` (lines 125–132)

```cpp
float e = fabsf(ref[i] - got[i]) / (fabsf(ref[i]) + 1e-6f);
```

Absolute error doesn't tell you much when values vary over many orders of
magnitude. Relative error (`|got - ref| / |ref|`) normalises by the expected
magnitude. The `+ 1e-6f` prevents division by zero for elements that are
exactly zero in the reference.

We compare our results against cuBLAS (not a CPU reference) because cuBLAS
itself uses fp32 internally for this call, so the bit patterns should match
closely. Threshold of 1e-3 (0.1%) is generous enough to pass.

---

## `bench.py`

### Why Python for GPU benchmarking?

PyTorch wraps cuBLAS. Calling `torch.matmul(a32, b32)` on CUDA tensors
calls `cublasSgemm` internally for fp32 and `cublasHgemm` (or tensor core
routines) for fp16. This lets you benchmark the same library code as matmul.cu
but from a ten-line Python script.

### Triton — what it is (lines 36–88)

Triton is a Python-embedded DSL (domain-specific language) for writing GPU
kernels. Instead of thinking in individual threads, you think in **tiles**:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for _ in range(0, K, BLOCK_K):
    a = tl.load(A, ...)   # load a tile of A
    b = tl.load(B, ...)   # load a tile of B
    acc += tl.dot(a, b)   # matrix multiply the tiles
```

`tl.dot()` compiles to the same WMMA tensor-core instructions that cuBLAS
uses. The Triton compiler handles the warp layout, register allocation, and
shared memory management automatically. Result: Triton fp32 performance is
close to cuBLAS fp32 despite being ~40 lines of Python.

### `tl.constexpr` (lines 43–45)

```python
BLOCK_M: tl.constexpr,
BLOCK_N: tl.constexpr,
BLOCK_K: tl.constexpr,
```

Tile dimensions marked `constexpr` become compile-time constants. The Triton
compiler specializes the kernel code for these exact values — it can unroll
loops, allocate the exact right amount of shared memory, and optimise
memory access patterns at compile time. Calling the kernel with different
block sizes generates different machine code.

### `tl.program_id` (lines 48–49)

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

In Triton, each kernel invocation is called a **program**. `program_id(0)`
is the index in the first grid dimension — the tile row. `program_id(1)` is
the tile column. The grid (set in `triton_matmul`) has `ceil(M/64)` × `ceil(N/64)`
programs, one per output tile.

### fp16 vs fp32 speedup (last few lines)

On RTX 3060 (Ampere, sm_86), fp16 matmul is typically 5–8× faster than fp32
matmul at large sizes. The reason: Ampere's 3rd-gen Tensor Cores execute
`16×16×16` fp16 MACs in a single clock cycle. fp32 uses the regular CUDA
cores in a scalar pipeline. Both have the same number of cores but Tensor
Core throughput is much higher.

The benchmark prints the `fp16/fp32` ratio so you can see this directly.

---

## `profile.sh`

### What `ncu` is

`ncu` (Nsight Compute) is the **hardware performance counter** profiler.
While the CUDA events in GpuTimer tell you wall-clock time, `ncu` reads
physical counters inside the chip:

- How many bytes actually moved over the DRAM bus?
- What fraction of warps were active each cycle?
- Were there shared memory bank conflicts?
- How many FFMA (fused floating-point multiply-add) instructions executed?

You pass `--metrics` followed by counter names. NVIDIA's counter naming
is verbose but systematic: `dram__bytes_read.sum` = "sum of bytes read
from DRAM across the entire kernel launch."

### Section 1 — vec_add bandwidth metrics

```bash
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum    # bytes loaded by LSU through L1
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum    # bytes stored by LSU through L1
dram__bytes_read.sum                             # actual DRAM reads
dram__bytes_write.sum                            # actual DRAM writes
sm__warps_active.avg.pct_of_peak_sustained_active  # occupancy %
l1tex__average_t_sectors_per_request_...ratio   # cache efficiency: 1.0 = fully coalesced
```

The last metric tells you if your accesses are coalesced. A 32-thread warp
requesting 32 consecutive floats should generate exactly one 128-byte cache
sector request. If threads are scattered, you might generate 32 sector
requests for the same number of reads — the ratio goes from 1.0 to 32.0.

### Section 2 & 3 — matmul occupancy

**For naive**:
```bash
sm__warps_active.avg.pct_of_peak_sustained_active   # occupancy
dram__bytes_read.sum                                 # should be enormous (N^3 redundant loads)
sm__sass_thread_inst_executed_op_ffma_pred_on.sum    # actual FMA instructions executed
```

The ratio of `ffma` to `dram bytes` reveals whether you're compute-bound or
memory-bound. For naive matmul, DRAM bytes is huge relative to FMA count —
memory bound.

**For tiled**, the extra metrics:
```bash
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  # write bank conflicts
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  # read bank conflicts
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct  # coalescing efficiency
```

**Shared memory bank conflicts**: the SM's shared memory is divided into 32
banks (one per thread in a warp). If all 32 threads in a warp access
different banks simultaneously, one cycle. If multiple threads hit the same
bank, they serialize. TILE=16 with row-major storage avoids conflicts because
consecutive threads in a warp access `As[ty][tx]` — different `tx` values =
different columns = different banks.

### Why `sudo` is often needed for `ncu`

Nsight Compute reads hardware performance counters, which are privileged
resources. Linux's `perf_event_paranoid` setting controls who can read them.
Setting it to 0 allows any process to read hardware counters without root:

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

This is a system-wide setting that persists until reboot.

---

## `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)
project(cuda_kernels CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 86)  # sm_86 = RTX 3060 / Ampere
```

`CMAKE_CUDA_ARCHITECTURES 86` tells `nvcc` to compile for `sm_86` — the
Ampere compute capability of the RTX 3060. Without this, nvcc compiles to
a generic PTX intermediate that is JIT-compiled on first run (slow), or to
an older arch that doesn't use Ampere-specific features like cp.async.

```cmake
set_target_properties(vec_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(vec_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler -O3>)
```

- `--use_fast_math`: enables `fmaf()` (fused multiply-add), approximate
  `__sinf/__cosf`, and flushes denormals to zero. Breaks IEEE strict
  compliance but is ~15% faster for fp32.
- `-Xcompiler -O3`: passes `-O3` to the host (C++) compiler. `nvcc`
  compiles CUDA device code itself; this flag governs the CPU-side code.

```cmake
target_link_libraries(matmul PRIVATE CUDA::cudart CUDA::cublas)
```

`CUDA::cublas` is a CMake imported target provided by the `FindCUDAToolkit`
module. It automatically adds the cuBLAS include path and links `libcublas.so`.
You don't need to find the .so manually.

---

## The roofline model — why tiled is only 1.5× faster than naive

This is the most important lesson in this chapter.

The roofline model plots two ceilings:

```
TFLOP/s
  │
  │  ●  cuBLAS (compute-bound)
3.7│
  │
  │
  │
0.5│         ●  tiled (memory-bound)
0.4│       ●  naive (memory-bound)
  └────────────────────────────── arithmetic intensity
```

For tiled matmul with TILE=16, each thread loads 16 elements from shared
memory per output element. The tile loads from DRAM are 1 per N/TILE = 256
elements. Arithmetic intensity ≈ N/TILE = 256 FLOPs per byte.

But the RTX 3060's compute throughput is only 9 TFLOP/s and its memory
bandwidth is 192 GB/s. The **ridge point** (where compute-bound meets
memory-bound) is at 9×10¹² / (192×10⁹) ≈ 47 FLOPs/byte. Our tiled kernel
at 256 FLOPs/byte should be compute-bound.

So why only 0.55 TFLOP/s?

**The answer: tile load latency**. Even though we only touch each tile once,
the 600-cycle DRAM latency for loading a 16×16 tile means the SM is stalled
for hundreds of cycles waiting for data to arrive in shared memory. With only
6 blocks per SM and each block in a `__syncthreads()` barrier during the load,
there aren't enough independent warps to hide that latency. The kernel is
**latency-bound**, not bandwidth-bound or compute-bound.

cuBLAS solves this with **double-buffering** (cp.async): while computing on
tile T's data already in registers, it asynchronously loads tile T+1 into
a second shared memory buffer. By the time compute finishes, the next tile
is already loaded. Latency is fully hidden.
