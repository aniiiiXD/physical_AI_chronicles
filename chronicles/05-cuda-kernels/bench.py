"""
bench.py — PyTorch / Triton matmul baseline comparison
RTX 3060 · sm_86

Run after compiling the CUDA kernels:
    python bench.py

Compares:
  - torch.matmul (calls cuBLAS, uses Tensor Cores in fp16)
  - torch.matmul fp32
  - a minimal Triton tiled matmul to show the DSL side
"""

import time
import torch
import triton
import triton.language as tl

DEVICE = "cuda"
torch.manual_seed(0)

# ── GPU info ──────────────────────────────────────────────────────────────────
props = torch.cuda.get_device_properties(0)
print(f"\n GPU: {props.name}")
print(f" SM count      : {props.multi_processor_count}")
print(f" VRAM          : {props.total_memory // 1024**3} GB")
print(f" Compute cap   : {props.major}.{props.minor}")
print(f" Peak FP32     : ~{props.multi_processor_count * 128 * 2 * 1.78 / 1000:.1f} TFLOP/s (est)")
print()

# ── Triton tiled matmul ───────────────────────────────────────────────────────
# Triton thinks in *tiles*, not individual threads.
# BLOCK_M/N/K define tile dimensions — the compiler maps these to warps.
# Each program (= one tile of C) accumulates partial sums over K-dimension tiles.

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # which tile of C does this program instance own?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row/col offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # accumulator — lives in registers for the whole loop
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # pointer arithmetic: start at top-left of each tile
    A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    for _ in range(0, K, BLOCK_K):
        # tl.load with mask handles boundaries — no if/else in kernel
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)          # maps to efficient WMMA/MMA instructions
        A  += BLOCK_K * stride_ak
        B  += BLOCK_K * stride_bk
        rk += BLOCK_K

    # write output tile
    C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c


# ── benchmark loop ────────────────────────────────────────────────────────────
def benchmark(fn, warmup=5, reps=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps  # ms


sizes = [1024, 2048, 4096]
print(f"{'N':>6}  {'torch fp32':>12}  {'torch fp16':>12}  {'triton fp32':>14}  {'fp16/fp32':>10}")
print("-" * 62)

for N in sizes:
    a32 = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    b32 = torch.randn(N, N, device=DEVICE, dtype=torch.float32)
    a16 = a32.half()
    b16 = b32.half()

    flops = 2.0 * N**3

    t32  = benchmark(lambda: torch.matmul(a32, b32))
    t16  = benchmark(lambda: torch.matmul(a16, b16))
    trit = benchmark(lambda: triton_matmul(a32, b32))

    gf32  = flops / (t32  * 1e-3) / 1e12
    gf16  = flops / (t16  * 1e-3) / 1e12
    gftri = flops / (trit * 1e-3) / 1e12
    ratio = gf16 / gf32

    print(f"{N:>6}  {gf32:>10.2f}T  {gf16:>10.2f}T  {gftri:>12.2f}T  {ratio:>9.1f}×")

print()
print("Notes:")
print("  fp16 > fp32 because cuBLAS uses Tensor Cores for fp16 (3rd gen on Ampere)")
print("  triton ≈ fp32 cuBLAS — same code path, compiler handles tiling")
print("  Your CUDA tiled kernel from matmul.cu will land between naive and cuBLAS")
print()
