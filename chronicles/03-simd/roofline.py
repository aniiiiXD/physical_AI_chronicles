#!/usr/bin/env python3
"""
Clean roofline plot.
Story: SIMD moves you UP the y-axis (more GFLOP/s at the same AI).
The roof tells you whether there's room above you.
"""

import subprocess, sys, re, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

# ── run benchmark ─────────────────────────────────────────────────────────────
print("Running ./simd_bench --csv …")
out = subprocess.run(["./simd_bench", "--csv"], capture_output=True, text=True)
lines = out.stdout.strip().splitlines()

hdr    = lines[0]
pk_gf   = float(re.search(r"peak_gflops=([\d.]+)", hdr).group(1))
pk_l1   = float(re.search(r"peak_l1_bw=([\d.]+)",  hdr).group(1))
pk_dram = float(re.search(r"peak_dram_bw=([\d.]+)", hdr).group(1))

rows = []
for line in lines[2:]:
    k, v, n, ai, gf = line.split(",")
    rows.append(dict(kernel=k, variant=v, n=int(n), ai=float(ai), gflops=float(gf)))

# ── keep only the L1 size (8 K) — fits in cache, shows pure compute story ─────
L1_N = 8 * 1024
data = [r for r in rows if r["n"] == L1_N]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xscale("log")
ax.set_yscale("log")

# ── roofline curves ───────────────────────────────────────────────────────────
AI = np.logspace(math.log10(0.06), math.log10(4), 600)

l1_roof   = np.minimum(pk_gf, pk_l1   * AI)
dram_roof = np.minimum(pk_gf, pk_dram * AI)

ax.fill_between(AI, dram_roof, pk_gf * 5,
                color="#f0f0f5", zorder=0)                # ceiling "above" space
ax.fill_between(AI, 0.5, dram_roof,
                color="#e8f4f8", alpha=0.6, zorder=0)     # below DRAM roof

ax.plot(AI, l1_roof,   color="#1a1a2e", lw=2.5, zorder=3,
        label=f"L1 cache roof  —  {pk_l1:.0f} GB/s")
ax.plot(AI, dram_roof, color="#666",    lw=2,   ls="--", zorder=3,
        label=f"DRAM roof  —  {pk_dram:.0f} GB/s")
ax.axhline(pk_gf, color="#c0392b", lw=1.5, ls=":", zorder=3,
           label=f"Peak compute  —  {pk_gf:.0f} GFLOP/s")

# ── ridge lines + labels ──────────────────────────────────────────────────────
ridge_l1   = pk_gf / pk_l1
ridge_dram = pk_gf / pk_dram
for rx, lbl, col in [(ridge_l1, "L1 ridge", "#1a1a2e"),
                     (ridge_dram, "DRAM ridge", "#666")]:
    ax.axvline(rx, color=col, lw=0.8, ls=":", alpha=0.5, zorder=2)
    ax.text(rx, pk_gf * 1.25, lbl, ha="center", va="bottom",
            fontsize=8, color=col,
            bbox=dict(fc="white", ec=col, pad=2, alpha=0.85))

# ── zone labels ───────────────────────────────────────────────────────────────
ax.text(0.065, pk_gf * 1.7,
        "MEMORY BOUND\n\nLimited by how fast\nRAM/cache can supply\ndata. Adding more\nFLOP/s won't help.",
        fontsize=9, color="#2471a3", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#2471a3", alpha=0.9))

ax.text(ridge_l1 * 1.15, pk_gf * 1.7,
        "COMPUTE BOUND\n\nLimited by the CPU's\narithmetic throughput.\nMore SIMD / unrolling\nCAN help here.",
        fontsize=9, color="#c0392b", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c0392b", alpha=0.9))

# ── plot data points ──────────────────────────────────────────────────────────
# Layout: for each kernel, show all 3 variants as a vertical group.
# Jitter on x-axis so they don't stack — scalar left, neon mid, neon4x right.

JITTER  = {"scalar": 0.87, "neon": 1.00, "neon4x": 1.15}
V_COLOR = {"scalar": "#e07b54", "neon": "#5b8dd9", "neon4x": "#2ecc71"}
V_LABEL = {"scalar": "Scalar", "neon": "NEON 4-wide", "neon4x": "NEON 4× unrolled"}
KERNEL  = {"dot": ("Dot\nproduct", "o", 0), "saxpy": ("SAXPY", "s", 0), "relu": ("ReLU", "^", 0)}

# Group by kernel → collect the three variant points
by_kernel = defaultdict(dict)
for r in data:
    by_kernel[r["kernel"]][r["variant"]] = r

for kname, variants in by_kernel.items():
    kname_str, mkr, _ = KERNEL[kname]
    base_ai = list(variants.values())[0]["ai"]

    # Draw vertical connector line between scalar and neon4x (shows SIMD "lift")
    ys = [variants[v]["gflops"] for v in ["scalar", "neon", "neon4x"] if v in variants]
    xs = [base_ai * JITTER[v] for v in ["scalar", "neon", "neon4x"] if v in variants]
    ax.plot([base_ai * JITTER["scalar"], base_ai * JITTER["neon4x"]],
            [variants["scalar"]["gflops"], variants["neon4x"]["gflops"]],
            color="#ccc", lw=1, ls="-", zorder=2)

    for vname, r in variants.items():
        xi = r["ai"] * JITTER[vname]
        yi = r["gflops"]
        col = V_COLOR[vname]
        ax.scatter(xi, yi, marker=mkr, color=col, s=150,
                   edgecolors="white", linewidths=1.5, zorder=5)

    # One kernel label at the neon4x point
    r4x = variants.get("neon4x") or variants.get("neon") or list(variants.values())[-1]
    x4x = r4x["ai"] * JITTER["neon4x"]

    # Theoretical ceiling for this kernel at L1 BW
    ceiling    = min(pk_gf, pk_l1 * r4x["ai"])
    gap_pct    = r4x["gflops"] / ceiling * 100
    bound_str  = "memory-bound" if pk_l1 * r4x["ai"] < pk_gf else "compute-bound"

    # Draw dashed gap line from neon4x up to the L1 roof
    ax.plot([x4x, x4x], [r4x["gflops"], ceiling],
            color=V_COLOR["neon4x"], lw=1, ls=":", alpha=0.6, zorder=3)
    ax.scatter(x4x, ceiling, marker="_", color=V_COLOR["neon4x"],
               s=100, zorder=4, lw=2, alpha=0.6)

    # Label
    label = (f"{kname_str}\n"
             f"AI = {r4x['ai']:.3f} FLOP/B\n"
             f"neon4x: {r4x['gflops']:.1f} GFLOP/s\n"
             f"= {gap_pct:.0f}% of L1 ceil\n"
             f"({bound_str})")

    offset_x = {"dot": 14, "saxpy": -115, "relu": 14}
    offset_y = {"dot":  5, "saxpy":    5, "relu": -55}
    ax.annotate(label,
                xy=(x4x, r4x["gflops"]),
                xytext=(offset_x[kname], offset_y[kname]),
                textcoords="offset points",
                fontsize=8.5, color="#222",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=V_COLOR["neon4x"], lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle="-|>", color=V_COLOR["neon4x"],
                                lw=1, mutation_scale=10))

# ── scalar dot annotation (shows how far behind scalar is) ──────────────────
sc_dot = by_kernel["dot"]["scalar"]
ax.annotate("scalar dot:\n4 GFLOP/s\n(13× below neon4x)",
            xy=(sc_dot["ai"] * JITTER["scalar"], sc_dot["gflops"]),
            xytext=(-65, -40), textcoords="offset points",
            fontsize=8, color=V_COLOR["scalar"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=V_COLOR["scalar"], alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=V_COLOR["scalar"],
                            lw=0.8, mutation_scale=8))

# ── legend ────────────────────────────────────────────────────────────────────
roof_leg = [
    Line2D([0],[0], color="#1a1a2e", lw=2.5,       label=f"L1 roofline  ({pk_l1:.0f} GB/s)"),
    Line2D([0],[0], color="#666",    lw=2, ls="--", label=f"DRAM roofline ({pk_dram:.0f} GB/s)"),
    Line2D([0],[0], color="#c0392b", lw=1.5, ls=":",label=f"Peak compute  ({pk_gf:.0f} GFLOP/s)"),
]
var_leg = [
    mpatches.Patch(color=V_COLOR["scalar"], label="Scalar"),
    mpatches.Patch(color=V_COLOR["neon"],   label="NEON  4-wide"),
    mpatches.Patch(color=V_COLOR["neon4x"], label="NEON  4× unrolled"),
]
ker_leg = [
    Line2D([0],[0], marker="o", color="#aaa", ls="none", ms=9, label="dot product"),
    Line2D([0],[0], marker="s", color="#aaa", ls="none", ms=9, label="saxpy"),
    Line2D([0],[0], marker="^", color="#aaa", ls="none", ms=9, label="relu"),
]
l1 = ax.legend(handles=roof_leg, loc="lower right", fontsize=9,
               title="Ceilings", title_fontsize=9, framealpha=0.95)
l2 = ax.legend(handles=var_leg,  loc="lower left",  fontsize=9,
               title="Variant",  title_fontsize=9, framealpha=0.95)
l3 = ax.legend(handles=ker_leg,  loc="lower left",  fontsize=9,
               title="Kernel",   title_fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.19, 0))
ax.add_artist(l1); ax.add_artist(l2)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.06, 3.5)
ax.set_ylim(0.8, pk_gf * 5)
ax.set_xlabel("Arithmetic Intensity   (FLOP / byte)   →   more compute-bound", fontsize=11)
ax.set_ylabel("Performance  (GFLOP/s)", fontsize=11)
ax.set_title("Roofline Model — 8 K array, fits in L1 cache\n"
             "Vertical grey lines: SIMD 'lifts' performance at the same arithmetic intensity",
             fontsize=12, pad=12)
ax.grid(True, which="both", ls=":", alpha=0.3)

plt.tight_layout()
plt.savefig("roofline.png", dpi=160, bbox_inches="tight")
print("Saved → roofline.png")
plt.show()
