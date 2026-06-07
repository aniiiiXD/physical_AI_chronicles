# SIMD Vectorisation — Line by Line Explainer

---

## What is SIMD?

**SIMD** = Single Instruction, Multiple Data.

Normally a CPU instruction operates on one value:

```
ADD  r0, r1        →  r0 = r0 + r1          (1 float)
```

A SIMD instruction operates on a *vector* of values at once:

```
FADD v0.4s, v1.4s  →  v0[0..3] = v0[0..3] + v1[0..3]   (4 floats)
```

Same clock cycle, 4× the work.  Apple Silicon's NEON registers are 128 bits
wide — they hold **4 × float32** or **2 × float64** simultaneously.

```
One 128-bit NEON register:
┌──────────┬──────────┬──────────┬──────────┐
│  lane 0  │  lane 1  │  lane 2  │  lane 3  │
│  float   │  float   │  float   │  float   │
└──────────┴──────────┴──────────┴──────────┘
```

---

## The include and the type

```cpp
#include <arm_neon.h>
```

This header declares every NEON intrinsic — functions that map directly to
single ARM assembly instructions.  On x86 you'd use `<immintrin.h>` for AVX.

```cpp
float32x4_t acc = vdupq_n_f32(0.0f);
```

`float32x4_t` — the C++ type for a 128-bit register of 4 floats.
`vdupq_n_f32(x)` — **dup**licate a scalar into all **q** (quad/128-bit)
lanes.  Result: `[0, 0, 0, 0]`.

**NEON naming convention**: `v` prefix, then operation, then `q` (128-bit)
or nothing (64-bit), then `_f32`/`_s32` etc. for element type.

---

## The intrinsics used, one by one

| Intrinsic | What it does | Assembly |
|-----------|-------------|----------|
| `vdupq_n_f32(x)` | Broadcast scalar x to 4 lanes | `DUP Vd.4S, Wn` |
| `vld1q_f32(ptr)` | Load 4 floats from memory | `LDR Qd, [Xn]` |
| `vst1q_f32(ptr, v)` | Store 4 floats to memory | `STR Qd, [Xn]` |
| `vfmaq_f32(a, b, c)` | a + b*c (fused mul-add) | `FMLA Vd.4S, Vn.4S, Vm.4S` |
| `vaddq_f32(a, b)` | a + b lane-wise | `FADD Vd.4S, Vn.4S, Vm.4S` |
| `vaddvq_f32(v)` | Sum all 4 lanes → scalar | `FADDP` chain |
| `vmaxq_f32(a, b)` | max(a, b) lane-wise | `FMAX Vd.4S, Vn.4S, Vm.4S` |

`vfmaq_f32(acc, va, vb)` does `acc = acc + va * vb` in a single instruction
with no rounding between the multiply and the add — more accurate AND faster
than separate multiply + add.

---

## Kernel 1 — Dot Product

### Why the scalar version is slow

```cpp
float sum = 0.0f;
for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];   // each iteration depends on the previous sum
```

Every iteration reads `sum`, updates it, writes it back.  There is a
**data dependency chain**: iteration i+1 cannot start until iteration i
finishes computing the new `sum`.  The FMA unit on Firestorm has ~4 cycle
latency, so the loop runs at one FMA every 4 cycles regardless of throughput.

### NEON version — 4 parallel lanes

```cpp
float32x4_t acc = vdupq_n_f32(0.0f);
for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vfmaq_f32(acc, va, vb);
}
float sum = vaddvq_f32(acc);
```

Still has a dependency chain on `acc`, but each iteration processes 4 elements
instead of 1 — immediate **4× throughput improvement** from just vectorising.

### NEON 4× unrolled — breaking the dependency chain

```cpp
float32x4_t acc0 = vdupq_n_f32(0.0f);
float32x4_t acc1 = vdupq_n_f32(0.0f);
float32x4_t acc2 = vdupq_n_f32(0.0f);
float32x4_t acc3 = vdupq_n_f32(0.0f);
for (; i + 16 <= n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
}
float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
```

`acc0..acc3` are **independent** of each other — the CPU can dispatch all four
FMAs to separate execution units simultaneously (**ILP: instruction-level
parallelism**).  The FMA latency is hidden because while acc0's result is
being computed, acc1/acc2/acc3 are already in flight.

Then the four accumulators are merged at the end:
- `vaddq_f32(acc0, acc1)` adds lane-by-lane
- The final `vaddvq_f32` horizontally sums the 4 lanes of the merged vector

**Result**: 13.5× speedup vs scalar at small N (in L1 cache).

---

## Kernel 2 — SAXPY

```cpp
vy = vfmaq_f32(vy, va, vx);   // vy += va * vx
vst1q_f32(y + i, vy);          // write back to y
```

Notice `va = vdupq_n_f32(alpha)` is **outside the loop** — computed once,
broadcast to all 4 lanes, reused every iteration.  The compiler would do the
same but it's explicit here.

### Why the speedup is ~1× (nearly zero)

```
Scalar SAXPY:  ~150 GB/s
NEON SAXPY:    ~155 GB/s
Speedup:       ~1.0×
```

SAXPY reads 2 arrays and writes 1 array per element = 12 bytes of memory per
element.  At N=16M that's 192 MB of data — the bottleneck is the memory bus,
not the CPU.

The CPU can issue NEON instructions faster than DRAM can supply operands.
SIMD instructions are sitting idle waiting for loads.  Making instructions
faster doesn't help when you're waiting for data.

The compiler's auto-vectoriser already generates NEON for scalar SAXPY
(because the loop has no dependencies between iterations), so manual NEON
doesn't add anything.

**Rule**: SIMD helps when you are **compute-bound**.  It does NOT help when
you are **memory-bandwidth-bound**.

---

## Kernel 3 — ReLU

```cpp
float32x4_t zero = vdupq_n_f32(0.0f);
vst1q_f32(y + i, vmaxq_f32(vx, zero));
```

`vmaxq_f32(a, b)` computes `max(a[i], b[i])` for each lane in one instruction.
No branch, no comparison + select — just a single hardware max.

The scalar version has a branch:
```cpp
y[i] = x[i] > 0.0f ? x[i] : 0.0f;
```

But the compiler will typically compile this to a `FMAX` instruction anyway,
so the speedup is small (~1.1×).  Again this is memory-bandwidth-bound at
large N.

---

## Reading the results — the key lesson

```
KERNEL 1 (Dot, compute-bound):
  8K    Scalar: 11 GB/s   NEON-4x: 157 GB/s   Speedup: 13.5×  ← SIMD wins big
  16M   Scalar: 16 GB/s   NEON-4x:  80 GB/s   Speedup:  4.9×  ← still good

KERNEL 2 (SAXPY, memory-bound):
  8K    Scalar: 295 GB/s  NEON-4x: 295 GB/s   Speedup:  1.0×  ← no gain
  16M   Scalar: 106 GB/s  NEON-4x: 104 GB/s   Speedup:  0.98× ← no gain
```

| Bound by | SIMD helps? | Why |
|----------|-------------|-----|
| Compute (FMA throughput) | **Yes — large** | More arithmetic per cycle |
| Memory bandwidth | **No** | Data supply is the bottleneck, not instruction throughput |

Dot product in L1 cache (8K) hits 157 GB/s — way above Apple M-series DRAM
bandwidth (~60–100 GB/s) — proving the data IS in cache.  SAXPY hitting
295 GB/s at 8K is also cache, but all three implementations hit the same
number because even scalar hits the bandwidth ceiling of the L1 cache.

---

## What `volatile float sink` does

```cpp
volatile float sink = 0;
double t_s = bench([&]{ sink = dot_scalar(a.data(), b.data(), n); }, reps);
```

Without `volatile`, the compiler can see that `dot_scalar`'s return value is
never used and delete the entire call.  `volatile` marks `sink` as something
that might be observed externally (like a hardware register), forcing the
store — and thus the computation — to actually happen.

---

## The `bench` template function

```cpp
template<typename F>
static double bench(F fn, int n_reps) {
    std::vector<double> times;
    for (int r = 0; r < n_reps; ++r) {
        double t0 = now_ns();
        fn();
        times.push_back(now_ns() - t0);
    }
    std::sort(times.begin(), times.end());
    return times[n_reps / 2];  // median
}
```

`template<typename F>` makes this work with any callable — a lambda, a
function pointer, a functor.  `F` is deduced automatically at the call site.

We use the **median** instead of the mean because OS context switches and
thermal throttling cause occasional large spikes.  Median is robust to those.

---

## Bandwidth formula

```cpp
static double gb_per_sec(long n, int bytes_per_elem, double elapsed_ns) {
    return (double)n * bytes_per_elem / elapsed_ns;
}
```

`n * bytes_per_elem` = total bytes touched.
Dividing by nanoseconds gives GB/s directly, because:
`bytes / nanoseconds = bytes / (seconds × 10⁻⁹) = (bytes × 10⁹) / seconds = GB/s`
(treating GB as 10⁹ bytes, not 2³⁰).

---

## Connection to the matmul result (02-matmul)

The dot product here and the inner loop of `matmul_blocked` are the same
operation.  If you added NEON intrinsics to the matmul inner kernel (the
`a * B[k*N+j]` line), you'd multiply the GFLOP/s by roughly the same factor
seen here — taking the ~23 GFLOP/s blocked result to potentially ~100+ GFLOP/s.
That is what BLAS libraries (OpenBLAS, Apple's Accelerate) do internally.
