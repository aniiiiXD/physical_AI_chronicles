"""
latency.py — end-to-end SmolVLA latency breakdown + control Hz

Measures each stage of the inference pipeline with CUDA events
(more accurate than time.perf_counter for GPU work).

Stages timed:
    1. image_prep   — PIL resize + ToTensor (CPU)
    2. vlm_encode   — SmolVLM2 vision + language encoding (GPU)
    3. diffusion    — action diffusion denoising loop (GPU)
    4. total        — full select_action call

Reports:
    - mean / p50 / p95 latency per stage
    - dominant operation
    - achievable control Hz

Run:
    source ~/lerobot-env/bin/activate
    python latency.py
"""

import time
import statistics
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

WARMUP_REPS = 5
BENCH_REPS  = 20

# ── CUDA event timer ──────────────────────────────────────────────────────────
class CudaTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self) -> float:
        self.e.record()
        torch.cuda.synchronize()
        return self.s.elapsed_time(self.e)  # ms

# ── load model ────────────────────────────────────────────────────────────────
print("Loading SmolVLA...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").eval()

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
)
task = "pick up the red block and place it in the box"
enc  = tokenizer(task, return_tensors="pt", padding=True)
lang_tokens = enc["input_ids"].cuda()
lang_mask   = enc["attention_mask"].bool().cuda()

# synthetic scene with red block
img_pil = Image.new("RGB", (256, 256), (180, 180, 175))
ImageDraw.Draw(img_pil).rectangle([100, 100, 156, 156], fill=(210, 45, 45))
img_t = TF.to_tensor(img_pil).unsqueeze(0).cuda()
state = torch.tensor([[0.0, -0.3, 0.6, 0.0, 0.9, 0.0]]).cuda()

batch = {
    "observation.images.camera1":        img_t,
    "observation.images.camera2":        img_t,
    "observation.images.camera3":        img_t,
    "observation.state":                 state,
    "observation.language.tokens":       lang_tokens,
    "observation.language.attention_mask": lang_mask,
}

# ── hook into SmolVLA internals to time sub-stages ────────────────────────────
stage_times = {"image_prep": [], "vlm_encode": [], "diffusion": [], "total": []}
_hooks = []

def _hook_vlm_start(module, input):
    module._t = CudaTimer(); module._t.start()

def _hook_vlm_end(module, input, output):
    stage_times["vlm_encode"].append(module._t.stop())

# try to attach hooks to the VLM and action model
try:
    vlm = policy.model.vlm_with_expert
    _hooks.append(vlm.register_forward_pre_hook(_hook_vlm_start))
    _hooks.append(vlm.register_forward_hook(_hook_vlm_end))
except AttributeError:
    pass  # model structure different — will rely on total time only

# ── warmup ────────────────────────────────────────────────────────────────────
print(f"Warming up ({WARMUP_REPS} passes)...")
with torch.no_grad():
    for _ in range(WARMUP_REPS):
        policy.reset(); policy.select_action(batch)
torch.cuda.synchronize()

# ── benchmark ─────────────────────────────────────────────────────────────────
print(f"Benchmarking ({BENCH_REPS} passes)...\n")
total_timer = CudaTimer()

with torch.no_grad():
    for _ in range(BENCH_REPS):
        # CPU image prep timing
        t0_cpu = time.perf_counter()
        _ = TF.to_tensor(img_pil.resize((256, 256)))
        stage_times["image_prep"].append((time.perf_counter() - t0_cpu) * 1000)

        # full forward pass
        policy.reset()
        total_timer.start()
        action = policy.select_action(batch)
        stage_times["total"].append(total_timer.stop())

# estimate diffusion time = total − vlm_encode
for i in range(BENCH_REPS):
    if len(stage_times["vlm_encode"]) > i:
        diff = stage_times["total"][i] - stage_times["vlm_encode"][i]
        stage_times["diffusion"].append(max(diff, 0))

# remove hooks
for h in _hooks: h.remove()

# ── report ────────────────────────────────────────────────────────────────────
def fmt(samples):
    if not samples:
        return "  n/a (hook unavailable)"
    s = sorted(samples)
    return (f"  mean={statistics.mean(s):7.1f} ms  "
            f"p50={statistics.median(s):7.1f} ms  "
            f"p95={s[int(len(s)*0.95)]:7.1f} ms")

print("═" * 60)
print("  SmolVLA forward-pass latency  |  RTX 3060")
print("═" * 60)
print(f"  image_prep (CPU):  {fmt(stage_times['image_prep'])}")
print(f"  vlm_encode (GPU):  {fmt(stage_times['vlm_encode'])}")
print(f"  diffusion  (GPU):  {fmt(stage_times['diffusion'])}")
print(f"  ── TOTAL ─────────{fmt(stage_times['total'])}")
print()

total_p50 = statistics.median(sorted(stage_times["total"]))
hz = 1000.0 / total_p50

# identify dominant stage
candidates = {k: statistics.mean(v) for k, v in stage_times.items() if v and k != "total"}
dominant = max(candidates, key=candidates.get)

print(f"  Control Hz  : {hz:.1f} Hz  ({total_p50:.0f} ms per action)")
print(f"  Dominant op : {dominant}  ({candidates.get(dominant,0):.1f} ms)")
print()
print("  Interpretation:")
if hz >= 30:
    print("  ≥30 Hz — real-time control (industrial arms run at 100-1000 Hz)")
elif hz >= 10:
    print("  10–30 Hz — acceptable for slow manipulation tasks")
elif hz >= 5:
    print("  5–10 Hz — borderline; suitable for pick-and-place, not dexterous tasks")
else:
    print("  <5 Hz — too slow for real-time control without action chunking")
print()
print("  Note: SmolVLA uses action chunking — it predicts N future actions")
print("  at once and executes them open-loop. Effective Hz = chunk_size × Hz.")
print("═" * 60)
