// SIMD vectorisation benchmark — ARM NEON (Apple Silicon)
// Three classic hot loops, each in three flavours:
//   scalar      — plain C++, let the compiler do whatever it wants
//   neon        — manual NEON intrinsics, 4 floats per instruction
//   neon_4x     — same but 4× loop-unrolled (16 floats per iteration)

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <arm_neon.h>   // NEON intrinsics header — ARM equivalent of <immintrin.h>

// ─── prevent the compiler from eliminating result writes ─────────────────────
// Marking outputs `volatile` forces the store to actually happen so the
// compiler cannot see the loop as dead code and delete it.
// We only do this on the *output* pointer, not inputs — inputs are unchanged.

// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 1 — Dot product:  sum = Σ a[i] * b[i]
//
// Memory pattern: two reads per element, no writes (sum lives in a register).
// For large N this is memory-bandwidth-bound; for small N (fits in cache)
// it is compute-bound and shows raw FMA throughput.
// ═════════════════════════════════════════════════════════════════════════════

float dot_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

float dot_neon(const float* a, const float* b, int n) {
    // float32x4_t — NEON vector type: holds 4 float32 values side by side
    // in a 128-bit register (Q register on ARM).
    float32x4_t acc = vdupq_n_f32(0.0f);   // broadcast 0.0f to all 4 lanes
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i); // load 4 floats from a[i..i+3]
        float32x4_t vb = vld1q_f32(b + i); // load 4 floats from b[i..i+3]
        acc = vfmaq_f32(acc, va, vb);       // acc += va * vb  (fused mul-add)
    }
    // vaddvq_f32 — horizontal add: sums the 4 lanes into one scalar
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) sum += a[i] * b[i]; // scalar tail for leftover elements
    return sum;
}

float dot_neon_4x(const float* a, const float* b, int n) {
    // Four independent accumulators let the CPU execute 4 FMAs in flight
    // simultaneously (instruction-level parallelism, ILP).
    // Each Firestorm FMA unit has ~4 cycle latency — with one accumulator
    // the loop stalls waiting for the previous result; with 4 we keep it fed.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    // Merge the 4 accumulators pairwise, then reduce to scalar
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 2 — SAXPY:  y[i] = alpha * x[i] + y[i]
//
// "Single-precision A times X Plus Y" — a BLAS Level 1 staple.
// Memory pattern: 2 reads + 1 write per element.  Compute is 1 FMA per element.
// On large arrays this is dominated by memory bandwidth, not arithmetic.
// ═════════════════════════════════════════════════════════════════════════════

void saxpy_scalar(float alpha, const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = alpha * x[i] + y[i];
}

void saxpy_neon(float alpha, const float* x, float* y, int n) {
    float32x4_t va = vdupq_n_f32(alpha); // broadcast alpha to all 4 lanes once
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vfmaq_f32(vy, va, vx);         // vy = vy + va*vx
        vst1q_f32(y + i, vy);               // store 4 results back to y
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

void saxpy_neon_4x(float alpha, const float* x, float* y, int n) {
    float32x4_t va = vdupq_n_f32(alpha);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        // Load 4 vectors of x and y, compute 4 FMAs, store 4 results.
        // The CPU can pipeline the loads of the next 16 elements while
        // writing back the current 16.
        float32x4_t vy0 = vld1q_f32(y+i);    vy0 = vfmaq_f32(vy0, va, vld1q_f32(x+i));
        float32x4_t vy1 = vld1q_f32(y+i+4);  vy1 = vfmaq_f32(vy1, va, vld1q_f32(x+i+4));
        float32x4_t vy2 = vld1q_f32(y+i+8);  vy2 = vfmaq_f32(vy2, va, vld1q_f32(x+i+8));
        float32x4_t vy3 = vld1q_f32(y+i+12); vy3 = vfmaq_f32(vy3, va, vld1q_f32(x+i+12));
        vst1q_f32(y+i,    vy0);
        vst1q_f32(y+i+4,  vy1);
        vst1q_f32(y+i+8,  vy2);
        vst1q_f32(y+i+12, vy3);
    }
    for (; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 3 — ReLU:  y[i] = max(0, x[i])
//
// Used in every neural network. No arithmetic — just a comparison and select.
// Shows SIMD for non-FMA operations.
// Memory: 1 read + 1 write. Purely memory-bandwidth-bound.
// ═════════════════════════════════════════════════════════════════════════════

void relu_scalar(const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void relu_neon(const float* x, float* y, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        // vmaxq_f32: lane-wise max(a, b) — no branch, single instruction
        vst1q_f32(y + i, vmaxq_f32(vx, zero));
    }
    for (; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void relu_neon_4x(const float* x, float* y, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(y+i,    vmaxq_f32(vld1q_f32(x+i),    zero));
        vst1q_f32(y+i+4,  vmaxq_f32(vld1q_f32(x+i+4),  zero));
        vst1q_f32(y+i+8,  vmaxq_f32(vld1q_f32(x+i+8),  zero));
        vst1q_f32(y+i+12, vmaxq_f32(vld1q_f32(x+i+12), zero));
    }
    for (; i < n; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// ─── timing ──────────────────────────────────────────────────────────────────
static double now_ns() {
    return std::chrono::duration<double, std::nano>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

// ─── bandwidth in GB/s ───────────────────────────────────────────────────────
// bytes_per_elem: how many float reads+writes does each element trigger?
// dot:   2 reads  → 2*4 = 8 bytes/elem
// saxpy: 2 reads + 1 write → 3*4 = 12 bytes/elem
// relu:  1 read  + 1 write → 2*4 =  8 bytes/elem
static double gb_per_sec(long n, int bytes_per_elem, double elapsed_ns) {
    return (double)n * bytes_per_elem / elapsed_ns; // ns cancels with 1e9, giving GB/s
}

// ─── run a kernel N_REPS times, return median time in ns ─────────────────────
template<typename F>
static double bench(F fn, int n_reps) {
    std::vector<double> times;
    times.reserve(n_reps);
    for (int r = 0; r < n_reps; ++r) {
        double t0 = now_ns();
        fn();
        times.push_back(now_ns() - t0);
    }
    std::sort(times.begin(), times.end());
    return times[n_reps / 2]; // median — robust against OS jitter spikes
}

// ─── print a separator row ───────────────────────────────────────────────────
static void sep(int w = 88) { std::cout << std::string(w, '-') << '\n'; }

// ─── peak compute (no memory traffic) ────────────────────────────────────────
// 8 independent NEON FMA chains feed all execution units simultaneously.
// No loads/stores — pure arithmetic throughput.
static double peak_gflops_measured() {
    float32x4_t a = vdupq_n_f32(1.0001f);
    float32x4_t b = vdupq_n_f32(0.9999f);
    float32x4_t s0=a,s1=a,s2=a,s3=a,s4=a,s5=a,s6=a,s7=a;
    const int ITERS = 1 << 26;
    double t0 = now_ns();
    for (int i = 0; i < ITERS; ++i) {
        s0=vfmaq_f32(s0,a,b); s1=vfmaq_f32(s1,a,b);
        s2=vfmaq_f32(s2,a,b); s3=vfmaq_f32(s3,a,b);
        s4=vfmaq_f32(s4,b,a); s5=vfmaq_f32(s5,b,a);
        s6=vfmaq_f32(s6,b,a); s7=vfmaq_f32(s7,b,a);
    }
    double elapsed_s = (now_ns() - t0) * 1e-9;
    volatile float sink = vaddvq_f32(vaddq_f32(vaddq_f32(s0,s1),vaddq_f32(s2,s3)));
    (void)sink;
    return 8.0 * 4 * 2 * ITERS / elapsed_s / 1e9;  // 8 vectors × 4 lanes × 2 FLOPs(FMA)
}

// ─── CSV output mode ──────────────────────────────────────────────────────────
// Emits roofline-ready CSV: kernel, variant, N, arithmetic_intensity, gflops
static void run_csv() {
    const int sizes[] = { 8*1024, 64*1024, 2*1024*1024, 16*1024*1024 };
    std::mt19937 rng(0xbeef);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Measure peaks
    double pk_gf  = peak_gflops_measured();
    std::vector<float> bx(16*1024*1024), by(16*1024*1024);
    for (auto& v : bx) v = dist(rng);
    for (auto& v : by) v = dist(rng);
    double pk_l1   = gb_per_sec(8*1024,        12, bench([&]{ saxpy_neon_4x(1.5f,bx.data(),by.data(),8*1024);        },20));
    double pk_dram = gb_per_sec(16*1024*1024,   12, bench([&]{ saxpy_neon_4x(1.5f,bx.data(),by.data(),16*1024*1024); },10));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "# peak_gflops=" << pk_gf
              << " peak_l1_bw="   << pk_l1
              << " peak_dram_bw=" << pk_dram << "\n";
    std::cout << "kernel,variant,n,ai,gflops\n";

    // emit one CSV row: flop = FLOPs per element, bpe = bytes per element
    auto emit = [&](const char* k, const char* v, int n, int flop, int bpe, double t_ns) {
        double ai = (double)flop / bpe;
        double gf = (double)flop * n / (t_ns * 1e-9) / 1e9;
        std::cout << k << "," << v << "," << n << "," << ai << "," << gf << "\n";
    };

    for (int n : sizes) {
        int reps = std::max(5, 1 << 27 >> (int)std::log2(n));
        std::vector<float> a(n),b(n),x(n),ys(n),yn(n),y4(n),rs(n),rn(n),r4(n);
        for (auto& w : a) w=dist(rng); for (auto& w : b) w=dist(rng);
        for (auto& w : x) w=dist(rng); for (auto& w : ys) w=dist(rng);
        std::copy(ys.begin(),ys.end(),yn.begin()); std::copy(ys.begin(),ys.end(),y4.begin());

        volatile float sk=0;
        emit("dot","scalar",n,2,8, bench([&]{sk=dot_scalar (a.data(),b.data(),n);},reps));
        emit("dot","neon",  n,2,8, bench([&]{sk=dot_neon   (a.data(),b.data(),n);},reps));
        emit("dot","neon4x",n,2,8, bench([&]{sk=dot_neon_4x(a.data(),b.data(),n);},reps));
        (void)sk;
        float al=2.5f;
        emit("saxpy","scalar",n,2,12,bench([&]{saxpy_scalar (al,x.data(),ys.data(),n);},reps));
        emit("saxpy","neon",  n,2,12,bench([&]{saxpy_neon   (al,x.data(),yn.data(),n);},reps));
        emit("saxpy","neon4x",n,2,12,bench([&]{saxpy_neon_4x(al,x.data(),y4.data(),n);},reps));
        emit("relu","scalar",n,1,8, bench([&]{relu_scalar (x.data(),rs.data(),n);},reps));
        emit("relu","neon",  n,1,8, bench([&]{relu_neon   (x.data(),rn.data(),n);},reps));
        emit("relu","neon4x",n,1,8, bench([&]{relu_neon_4x(x.data(),r4.data(),n);},reps));
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--csv") { run_csv(); return 0; }

    // Array sizes: sweep from in-cache to RAM
    const int sizes[] = {
        8   * 1024,        //  32 KB — fits in L1
        64  * 1024,        // 256 KB — fits in L2
        2   * 1024*1024,   //   8 MB — fits in L3
        16  * 1024*1024,   //  64 MB — RAM
    };

    std::mt19937 rng(0xbeef);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Header
    auto hdr = [](const std::string& name) {
        std::cout << '\n' << name << '\n';
        std::cout << std::left
                  << std::setw(10) << "N"
                  << std::setw(14) << "Scalar(GB/s)"
                  << std::setw(14) << "NEON(GB/s)"
                  << std::setw(16) << "NEON-4x(GB/s)"
                  << std::setw(12) << "Speedup-4x"
                  << '\n';
        sep();
    };

    // ── DOT PRODUCT ──────────────────────────────────────────────────────────
    hdr("KERNEL 1: Dot Product   sum = Σ a[i]*b[i]   (2 reads/elem, 1 FMA/elem)");
    for (int n : sizes) {
        std::vector<float> a(n), b(n);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        int reps = std::max(5, 1 << 27 >> (int)std::log2(n));

        volatile float sink = 0;   // volatile sink: force result to be used

        double t_s  = bench([&]{ sink = dot_scalar(a.data(), b.data(), n); }, reps);
        double t_n  = bench([&]{ sink = dot_neon  (a.data(), b.data(), n); }, reps);
        double t_4x = bench([&]{ sink = dot_neon_4x(a.data(), b.data(), n); }, reps);
        (void)sink;

        std::string sz = (n >= 1<<20) ? std::to_string(n>>20)+"M" : std::to_string(n>>10)+"K";
        std::cout << std::left << std::fixed << std::setprecision(1)
                  << std::setw(10) << sz
                  << std::setw(14) << gb_per_sec(n, 8, t_s)
                  << std::setw(14) << gb_per_sec(n, 8, t_n)
                  << std::setw(16) << gb_per_sec(n, 8, t_4x)
                  << std::setprecision(2) << std::setw(8) << (t_s / t_4x) << "x\n";
    }

    // ── SAXPY ─────────────────────────────────────────────────────────────────
    hdr("\nKERNEL 2: SAXPY   y[i] = alpha*x[i] + y[i]   (2 reads + 1 write/elem)");
    for (int n : sizes) {
        std::vector<float> x(n), y_s(n), y_n(n), y_4x(n);
        for (auto& v : x)   v = dist(rng);
        for (auto& v : y_s) v = dist(rng);
        std::copy(y_s.begin(), y_s.end(), y_n.begin());
        std::copy(y_s.begin(), y_s.end(), y_4x.begin());

        int reps = std::max(5, 1 << 27 >> (int)std::log2(n));
        float alpha = 2.5f;

        double t_s  = bench([&]{ saxpy_scalar(alpha, x.data(), y_s.data(),  n); }, reps);
        double t_n  = bench([&]{ saxpy_neon  (alpha, x.data(), y_n.data(),  n); }, reps);
        double t_4x = bench([&]{ saxpy_neon_4x(alpha, x.data(), y_4x.data(),n); }, reps);

        std::string sz = (n >= 1<<20) ? std::to_string(n>>20)+"M" : std::to_string(n>>10)+"K";
        std::cout << std::left << std::fixed << std::setprecision(1)
                  << std::setw(10) << sz
                  << std::setw(14) << gb_per_sec(n, 12, t_s)
                  << std::setw(14) << gb_per_sec(n, 12, t_n)
                  << std::setw(16) << gb_per_sec(n, 12, t_4x)
                  << std::setprecision(2) << std::setw(8) << (t_s / t_4x) << "x\n";
    }

    // ── RELU ─────────────────────────────────────────────────────────────────
    hdr("\nKERNEL 3: ReLU   y[i] = max(0, x[i])   (1 read + 1 write/elem)");
    for (int n : sizes) {
        std::vector<float> x(n), y_s(n), y_n(n), y_4x(n);
        for (auto& v : x) v = dist(rng);

        int reps = std::max(5, 1 << 27 >> (int)std::log2(n));

        double t_s  = bench([&]{ relu_scalar (x.data(), y_s.data(),  n); }, reps);
        double t_n  = bench([&]{ relu_neon   (x.data(), y_n.data(),  n); }, reps);
        double t_4x = bench([&]{ relu_neon_4x(x.data(), y_4x.data(), n); }, reps);

        std::string sz = (n >= 1<<20) ? std::to_string(n>>20)+"M" : std::to_string(n>>10)+"K";
        std::cout << std::left << std::fixed << std::setprecision(1)
                  << std::setw(10) << sz
                  << std::setw(14) << gb_per_sec(n, 8, t_s)
                  << std::setw(14) << gb_per_sec(n, 8, t_n)
                  << std::setw(16) << gb_per_sec(n, 8, t_4x)
                  << std::setprecision(2) << std::setw(8) << (t_s / t_4x) << "x\n";
    }

    std::cout << '\n';
}
