#!/usr/bin/env bash
# profile.sh — Nsight Compute profiling for RTX 3060
#
# Requirements:
#   sudo ncu  (ncu needs root on most Linux setups, or:)
#   sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
#   sudo sh -c 'echo 0 > /proc/sys/kernel/yama/ptrace_scope'
#
# Usage:
#   bash profile.sh

set -e
BUILD="./build"

echo ""
echo "══════════════════════════════════════════════"
echo "  1 / 3 — vec_add bandwidth profile"
echo "══════════════════════════════════════════════"
ncu \
  --metrics \
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
  --target-processes all \
  "$BUILD/vec_add"

echo ""
echo "══════════════════════════════════════════════"
echo "  2 / 3 — matmul_naive occupancy + DRAM"
echo "══════════════════════════════════════════════"
ncu \
  --kernel-name "matmul_naive" \
  --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
  "$BUILD/matmul"

echo ""
echo "══════════════════════════════════════════════"
echo "  3 / 3 — matmul_tiled occupancy + shared mem"
echo "══════════════════════════════════════════════"
ncu \
  --kernel-name "matmul_tiled" \
  --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
  "$BUILD/matmul"

echo ""
echo "══════════════════════════════════════════════"
echo "  What to look for:"
echo ""
echo "  vec_add:"
echo "    dram__bytes_read.sum ≈ 512 MB (2 × 256 MB input arrays)"
echo "    achieved BW / 360 GB/s = efficiency"
echo ""
echo "  matmul_naive:"
echo "    sm__warps_active ≈ 30–50%  (low — small tile = poor occupancy)"
echo "    dram__bytes_read huge      (N^3 redundant loads)"
echo ""
echo "  matmul_tiled:"
echo "    sm__warps_active ≈ 70–90%  (TILE=32, full warp utilisation)"
echo "    dram__bytes_read ≈ N^2×2×4 (each tile loaded once)"
echo "    bank conflicts = 0         (TILE=32 avoids stride conflicts)"
echo "══════════════════════════════════════════════"
echo ""

# ── quick nsys timeline (optional, creates a .nsys-rep file) ─────────────────
# nsys profile --stats=true -o matmul_timeline ./build/matmul
