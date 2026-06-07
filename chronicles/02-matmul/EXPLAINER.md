# Cache-Blocked Matrix Multiply — Line by Line Explainer

---

## Big picture: why does matmul care about cache?

Matrix multiply computes C = A × B where each element is:

```
C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + … + A[i][N-1]*B[N-1][j]
```

Written as a triple loop this is O(N³) arithmetic operations.  For N=1024
that's ~2 billion multiplications + additions.  The question isn't "can we
do fewer flops?" — we can't.  The question is "can we do those flops
without going to RAM each time?"

### Row-major memory layout

Both matrices are stored flat in a `std::vector<float>` in **row-major** order:

```
Matrix (3×3):          Memory:
  a b c               [a b c | d e f | g h i]
  d e f                ← row 0 →← row 1 →← row 2 →
  g h i
```

Element `[i][j]` lives at index `i*N + j`.

Accessing a whole row is fast — elements are adjacent in memory, so one
cache line (64 bytes = 16 floats) loads many elements at once.  Accessing a
whole *column* is slow — elements are N floats apart, so each access hits a
different cache line.

### Why the naïve version is slow

```
for i:
  for j:
    for k:
      C[i][j] += A[i][k] * B[k][j]   ← B is column access (stride N)
```

`B[k][j]` steps through column `j` of B — each successive `k` jumps N*4
bytes forward in memory.  For N=1024 that's 4096 bytes between accesses —
much larger than a cache line.  Every access to B is a **cache miss**.

---

## The includes and types

```cpp
#include <vector>    // std::vector — heap-allocated resizable array
#include <chrono>    // high-resolution timer
#include <cmath>     // std::fabs — floating-point absolute value
#include <cstring>   // std::memset — zero a block of memory
#include <random>    // std::mt19937, std::uniform_real_distribution
#include <algorithm> // std::min, std::max
```

`float` (32-bit) is used instead of `double` (64-bit) because:
- SIMD registers can hold twice as many floats per instruction.
- Most ML/HPC code uses float for matmul.
- The cache footprint is half, so more fits in each cache level.

---

## `matmul_naive` — the baseline

```cpp
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}
```

**`const float*`** — raw pointer to float, `const` means we promise not to
modify through it.  Passing arrays as pointers avoids copying.

**`float sum = 0.0f`** — accumulate into a register variable rather than
writing back to `C` on every iteration.  The `f` suffix makes the literal
a `float` constant (without it, `0.0` is a `double`).

**`A[i*N + k]`** — row `i` of A, column `k`.  This walks along a row of A
(good — stride 1).

**`B[k*N + j]`** — row `k` of B, column `j`.  As `k` increments, this
jumps by N floats each step (bad — large stride, many cache misses).

---

## `matmul_blocked` — the optimised version

### The core idea: tiling

Instead of iterating `i` from 0 to N, `j` from 0 to N, `k` from 0 to N all
the way through, we break each dimension into tiles of size `BLOCK`:

```
i: [0..BLOCK) [BLOCK..2*BLOCK) … — tiles of rows of A and C
j: [0..BLOCK) [BLOCK..2*BLOCK) … — tiles of cols of B and C
k: [0..BLOCK) [BLOCK..2*BLOCK) … — tiles of "depth" (cols of A, rows of B)
```

At any moment we only work on three BLOCK×BLOCK sub-matrices — a tile of A,
a tile of B, a tile of C.  If BLOCK=64:

```
Working set = 3 tiles × 64×64 floats × 4 bytes = 49,152 bytes ≈ 48 KB
```

That fits comfortably in L2 cache (~256 KB).  We do BLOCK multiply-adds
per element before the tile is evicted — reusing each cache line BLOCK times
instead of once.

### Line by line

```cpp
std::memset(C, 0, N * N * sizeof(float));
```

Zero the output matrix.  We can't initialise to zero inside the tile loops
because multiple K-tiles all *accumulate* into the same C tile.

```cpp
for (int i0 = 0; i0 < N; i0 += BLOCK)    // tile start row
for (int j0 = 0; j0 < N; j0 += BLOCK)    // tile start col
for (int k0 = 0; k0 < N; k0 += BLOCK) {  // tile start depth
```

Three nested loops advance by `BLOCK` each step — these are the **tile
iterators**.  For N=256, BLOCK=64: each loop runs 256/64 = 4 times, so
we process 4×4×4 = 64 tile triples total.

```cpp
int i_end = std::min(i0 + BLOCK, N);
int j_end = std::min(j0 + BLOCK, N);
int k_end = std::min(k0 + BLOCK, N);
```

`std::min` clamps the tile boundary.  When N is not a multiple of BLOCK
(e.g. N=100, BLOCK=64), the last tile would overshoot — `min` prevents
out-of-bounds access.

```cpp
for (int i = i0; i < i_end; ++i)
for (int k = k0; k < k_end; ++k) {
    float a = A[i*N + k];            // ← hoisted A load
    for (int j = j0; j < j_end; ++j)
        C[i*N + j] += a * B[k*N + j];
}
```

**Inner micro-kernel — this is where the magic is.**

Notice the loop order is **i, k, j** — not i, j, k like the naïve version.

- `float a = A[i*N + k]` is hoisted outside the `j` loop.  One load serves
  `BLOCK` multiply-adds — the value sits in a CPU register the whole time.

- `B[k*N + j]`: as `j` increments by 1, this walks along row `k` of B
  (stride 1 — cache friendly!).  The entire tile row of B is loaded once
  and reused for all `i` values in the tile.

- `C[i*N + j]`: also stride 1 along the row — cache friendly.

Contrast with the naïve version's inner loop `B[k*N + j]` where `k` varied
and `j` was fixed — that was the column-stride disaster.  By swapping `k`
and `j` in the loop order we turned a column access into a row access.

---

## `now_ns` — timing

```cpp
static double now_ns() {
    return std::chrono::duration<double, std::nano>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}
```

`time_since_epoch()` returns the duration since the clock's epoch.
Converting to `duration<double, std::nano>` gives nanoseconds as a double.
We subtract two calls to get elapsed time.

---

## `matrices_close` — correctness check

```cpp
static bool matrices_close(const float* A, const float* B, int N, float tol = 1e-3f) {
    for (int i = 0; i < N * N; ++i)
        if (std::fabs(A[i] - B[i]) > tol * std::max(1.0f, std::fabs(A[i])))
            return false;
    return true;
}
```

Floating-point arithmetic is not associative.  The blocked version sums
partial products in a different order, so results differ slightly.  We use a
**relative tolerance** check: the error must be less than `tol` times the
magnitude of the reference value.  `std::max(1.0f, …)` avoids dividing by
zero near 0.

---

## `gflops` — measuring performance

```cpp
static double gflops(int N, double seconds) {
    return 2.0 * N * N * N / seconds / 1e9;
}
```

An N×N matmul does N³ multiplications and N³ additions = **2N³ FLOPs**.
Dividing by elapsed seconds and 1e9 gives **GFLOP/s** (giga floating-point
operations per second) — the standard metric for dense linear algebra.

Your results:
- Naïve: ~2 GFLOP/s  (bottlenecked by cache misses)
- Blocked B64: ~23 GFLOP/s  (11× speedup at N=1024)

A peak theoretical ~200+ GFLOP/s is possible with SIMD + multithreading
(BLAS libraries do this), but even pure scalar blocking gives ~10× here.

---

## `main` — the benchmark loop

```cpp
auto buf = std::make_unique<char[]>(MAX_BYTES);
```

Wait — this is from the other benchmark.  In this file we use:

```cpp
std::vector<float> A(sz), B(sz), C_naive(sz, 0), C_blocked(sz, 0);
```

`std::vector<float> A(sz)` allocates `sz` floats on the heap and
value-initialises them to 0.  `C_naive(sz, 0)` does the same with explicit
zero-fill.  Vectors automatically free memory when they go out of scope.

```cpp
int reps = std::max(1, 512 / (N + 1));
```

For small N, one iteration is microseconds — too noisy to time accurately.
We run `reps` repetitions and divide, getting a stable average.  For N=64,
reps ≈ 8.  For N=1024, reps = 1.

```cpp
for (int r = 0; r < reps; ++r) matmul_naive(A.data(), B.data(), C_naive.data(), N);
double naive_ms = (now_ns() - t0) / 1e6 / reps;
```

`A.data()` returns the raw `float*` pointer inside the vector — needed
because our functions take plain pointers, not vectors.  Dividing by `1e6`
converts nanoseconds to milliseconds; dividing by `reps` gives the per-run
average.

---

## How cache blocking maps to your CPU hierarchy

From the cache benchmark results in `01-cache-benchmark`:

| Level | Size    | Latency |
|-------|---------|---------|
| L1    | ≤128 KB | ~1 ns   |
| L2    | ≤16 MB  | ~3–9 ns |
| RAM   | >32 MB  | ~50–84 ns |

| BLOCK | Working set        | Fits in |
|-------|--------------------|---------|
| 32    | 3×32²×4 = 12 KB    | L1      |
| 64    | 3×64²×4 = 48 KB    | L2      |

B64 wins over B32 in most cases here because L2 bandwidth is higher than
what the inner loop can saturate at B32's working set size, and BLOCK=64
gives the inner loop more work to amortise loop overhead over.  On a machine
with a smaller L2, B32 might win.
