#!/usr/bin/env bash
# profile_nsight.sh — Nsight Systems timeline of one SmolVLA forward pass
#
# Captures a full GPU timeline: kernel launches, memory transfers,
# CUDA API calls, and CPU↔GPU synchronisation.
#
# Prerequisites:
#   nsys must be in PATH — ships with CUDA toolkit 12.x
#   which nsys   →  /usr/local/cuda/bin/nsys
#
# Run:
#   bash profile_nsight.sh
#
# Output:
#   smolvla_forward.nsys-rep   ← open in Nsight Systems GUI on any machine
#   smolvla_forward.sqlite     ← queryable with sqlite3 for CI/scripted analysis
#
# View results:
#   # On Ubuntu with GUI:
#   nsys-ui smolvla_forward.nsys-rep
#
#   # Headless text summary (shows top CUDA kernels by duration):
#   nsys stats smolvla_forward.nsys-rep

set -e

OUTFILE="smolvla_forward"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Nsight Systems — SmolVLA forward pass timeline"
echo "══════════════════════════════════════════════════"
echo ""

# Check nsys is available
if ! command -v nsys &> /dev/null; then
    echo "nsys not found. Add CUDA bin to PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

nsys profile \
    --output="$OUTFILE" \
    --force-overwrite=true \
    --trace=cuda,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stats=true \
    python "$SCRIPT_DIR/_forward_one.py"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Saved: $OUTFILE.nsys-rep"
echo ""
echo "  Top CUDA kernels (text summary):"
echo "══════════════════════════════════════════════════"
nsys stats --report cuda_gpu_kern_sum "$OUTFILE.nsys-rep" 2>/dev/null | head -40 || true
echo ""
echo "  To open the full GUI timeline:"
echo "    nsys-ui $OUTFILE.nsys-rep"
echo ""
echo "  What to look for:"
echo "    - Dominant kernel name → tells you which op is the bottleneck"
echo "    - cudaMemcpy gaps     → CPU↔GPU transfer overhead"
echo "    - Kernel occupancy    → how well the GPU is utilised"
echo "══════════════════════════════════════════════════"
