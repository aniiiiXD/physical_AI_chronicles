"""
latency.py — end-to-end SmolVLA latency breakdown + control Hz

Measures each stage of the inference pipeline with CUDA events
(more accurate than time.perf_counter for GPU work).

Stages timed:
    1. image_prep   — PIL resize + ToTensor (CPU)
    2. vlm_encode   — SmolVLM2 vision + language encoding (GPU)
    3. diffusion    — action diffusion denoising loop (GPU)
    4. total        — full select_action call

Sub-stage strategy (3 attempts in order):
    1. Monkey-patch vlm_with_expert.forward at instance level
    2. torch.profiler kernel table (one extra pass, shows op names)
    3. Wall-clock split using explicit synchronize points

Run:
    source ~/lerobot-env/bin/activate
    python latency.py
"""

import time
import statistics
import torch
import torch.profiler
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

img_pil = Image.new("RGB", (256, 256), (180, 180, 175))
ImageDraw.Draw(img_pil).rectangle([100, 100, 156, 156], fill=(210, 45, 45))
img_t = TF.to_tensor(img_pil).unsqueeze(0).cuda()
state = torch.tensor([[0.0, -0.3, 0.6, 0.0, 0.9, 0.0]]).cuda()

batch = {
    "observation.images.camera1":          img_t,
    "observation.images.camera2":          img_t,
    "observation.images.camera3":          img_t,
    "observation.state":                   state,
    "observation.language.tokens":         lang_tokens,
    "observation.language.attention_mask": lang_mask,
}

# ── Strategy 1: monkey-patch forward at instance level ────────────────────────
# forward hooks failed because select_action calls the VLM via a method other
# than __call__ (e.g. a custom encode/generate path). Patching .forward
# directly on the instance is called by nn.Module.__call__ regardless.
stage_times = {"image_prep": [], "vlm_encode": [], "diffusion": [], "total": []}
_vlm_event_pairs = []
_patched = False

_vlm_candidates = ["vlm_with_expert", "vlm", "smolvlm", "vision_language_model", "backbone"]
for _attr in _vlm_candidates:
    _vlm_mod = getattr(policy.model, _attr, None)
    if _vlm_mod is not None and isinstance(_vlm_mod, torch.nn.Module):
        _orig_forward = _vlm_mod.forward

        def _timed_forward(*args, _orig=_orig_forward, **kwargs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = _orig(*args, **kwargs)
            e.record()
            _vlm_event_pairs.append((s, e))
            return out

        _vlm_mod.forward = _timed_forward
        _patched = True
        print(f"  [latency] patched forward on policy.model.{_attr}")
        break

if not _patched:
    children = [n for n, _ in policy.model.named_children()]
    print(f"  [latency] patch unavailable. policy.model children: {children}")

# ── warmup ────────────────────────────────────────────────────────────────────
print(f"Warming up ({WARMUP_REPS} passes)...")
_vlm_event_pairs.clear()
with torch.no_grad():
    for _ in range(WARMUP_REPS):
        policy.reset()
        policy.select_action(batch)
torch.cuda.synchronize()

_patch_fired = len(_vlm_event_pairs) > 0
if _patched and not _patch_fired:
    print("  [latency] patched forward never called — falling back to profiler split")

# ── benchmark ─────────────────────────────────────────────────────────────────
print(f"Benchmarking ({BENCH_REPS} passes)...\n")
total_timer = CudaTimer()

with torch.no_grad():
    for _ in range(BENCH_REPS):
        t0_cpu = time.perf_counter()
        _ = TF.to_tensor(img_pil.resize((256, 256)))
        stage_times["image_prep"].append((time.perf_counter() - t0_cpu) * 1000)

        policy.reset()
        _vlm_event_pairs.clear()
        total_timer.start()
        action = policy.select_action(batch)
        total_ms = total_timer.stop()       # sync happens here
        stage_times["total"].append(total_ms)

        if _vlm_event_pairs:
            vlm_ms = sum(s.elapsed_time(e) for s, e in _vlm_event_pairs)
            stage_times["vlm_encode"].append(vlm_ms)
            stage_times["diffusion"].append(max(total_ms - vlm_ms, 0))

# ── Strategy 2: torch.profiler (one extra pass for op breakdown) ──────────────
profiler_table = None
if not stage_times["vlm_encode"]:
    print("  [latency] running torch.profiler for op breakdown (one extra pass)...")
    policy.reset()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            _ = policy.select_action(batch)
    torch.cuda.synchronize()

    events = sorted(prof.key_averages(), key=lambda x: x.cuda_time_total, reverse=True)
    profiler_table = events[:12]

# ── report ────────────────────────────────────────────────────────────────────
def fmt(samples):
    if not samples:
        return "  n/a"
    s = sorted(samples)
    return (f"  mean={statistics.mean(s):7.1f} ms  "
            f"p50={statistics.median(s):7.1f} ms  "
            f"p95={s[int(len(s)*0.95)]:7.1f} ms")

print("═" * 62)
print("  SmolVLA forward-pass latency  |  RTX 3060")
print("═" * 62)
print(f"  image_prep (CPU):  {fmt(stage_times['image_prep'])}")
print(f"  vlm_encode (GPU):  {fmt(stage_times['vlm_encode'])}")
print(f"  diffusion  (GPU):  {fmt(stage_times['diffusion'])}")
print(f"  ── TOTAL ─────────{fmt(stage_times['total'])}")
print()

total_p50 = statistics.median(sorted(stage_times["total"]))
hz = 1000.0 / total_p50

gpu_stages = {k: statistics.mean(v) for k, v in stage_times.items()
              if v and k not in ("total", "image_prep")}
if gpu_stages:
    dominant = max(gpu_stages, key=gpu_stages.get)
    dominant_ms = gpu_stages[dominant]
else:
    dominant = "GPU (vlm+diffusion combined)"
    dominant_ms = statistics.mean(stage_times["total"])

print(f"  Control Hz  : {hz:.1f} Hz  ({total_p50:.0f} ms per action)")
print(f"  Dominant op : {dominant}  ({dominant_ms:.1f} ms)")
print()
print("  Interpretation:")
if hz >= 30:
    print("  ≥30 Hz — real-time control")
elif hz >= 10:
    print("  10–30 Hz — acceptable for slow manipulation tasks")
elif hz >= 5:
    print("  5–10 Hz — borderline; suitable for pick-and-place, not dexterous tasks")
else:
    print("  <5 Hz — too slow for real-time control without action chunking")
print()
print("  SmolVLA uses action chunking: predicts N future actions at once,")
print("  executes them open-loop. Effective Hz = chunk_size × inference Hz.")
print("═" * 62)

# ── profiler table (if strategy 1 failed) ────────────────────────────────────
if profiler_table:
    print()
    print("  Top CUDA ops (torch.profiler, one pass):")
    print(f"  {'Op name':<50} {'CUDA ms':>8}")
    print("  " + "-" * 60)
    for ev in profiler_table:
        name = ev.key[:50]
        ms   = ev.cuda_time_total / 1000
        print(f"  {name:<50} {ms:8.2f}")
    print()
    print("  Look for: attention/matmul ops (VLM) vs repeated denoising ops (diffusion)")
