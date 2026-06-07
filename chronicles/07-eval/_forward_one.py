"""
_forward_one.py — single SmolVLA forward pass wrapped with CUDA profiler API
Called by profile_nsight.sh. Not meant to be run directly.

The torch.cuda.profiler.start/stop calls tell nsys exactly which region
to capture — no warmup noise, no model-load overhead in the timeline.
"""

import torch
import torch.cuda.profiler as profiler
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# load model
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").eval()
policy.reset()

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
task = "pick up the red block and place it in the box"
enc  = tokenizer(task, return_tensors="pt", padding=True)

# synthetic scene
img = Image.new("RGB", (256, 256), (180, 180, 175))
ImageDraw.Draw(img).rectangle([100, 100, 156, 156], fill=(210, 45, 45))
img_t = TF.to_tensor(img).unsqueeze(0).cuda()

batch = {
    "observation.images.camera1":        img_t,
    "observation.images.camera2":        img_t,
    "observation.images.camera3":        img_t,
    "observation.state":                 torch.tensor([[0.0,-0.3,0.6,0.0,0.9,0.0]]).cuda(),
    "observation.language.tokens":       enc["input_ids"].cuda(),
    "observation.language.attention_mask": enc["attention_mask"].bool().cuda(),
}

# warmup — 2 passes so CUDA JIT is done before profiling window opens
with torch.no_grad():
    for _ in range(2):
        policy.reset()
        policy.select_action(batch)
torch.cuda.synchronize()

# ── profiled region ───────────────────────────────────────────────────────────
policy.reset()
profiler.start()
with torch.no_grad():
    action = policy.select_action(batch)
torch.cuda.synchronize()
profiler.stop()
# ─────────────────────────────────────────────────────────────────────────────

print(f"Action: {action[0].cpu().numpy().round(3)}")
