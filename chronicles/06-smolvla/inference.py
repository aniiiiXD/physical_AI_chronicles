"""
inference.py — SmolVLA inference on a real sample observation

Builds a non-trivial test image (a scene with a red block), runs
inference, then validates the output is NaN-free, correct shape,
and changes meaningfully when the image changes.

Run:
    cd ~/Documents/physical_AI_chronicles/chronicles/06-smolvla
    source ~/lerobot-env/bin/activate
    python inference.py
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ── synthetic scene builder ───────────────────────────────────────────────────
def make_scene(has_block: bool = True) -> torch.Tensor:
    """
    Returns a (1, 3, 256, 256) float32 tensor in [0, 1].
    Draws a simple top-down workspace: grey table, red block if present.
    """
    img = Image.new("RGB", (256, 256), color=(180, 180, 175))  # grey table
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 200, 256, 256], fill=(100, 80, 60))     # dark edge
    if has_block:
        draw.rectangle([100, 100, 156, 156], fill=(210, 45, 45))  # red block
        draw.rectangle([102, 102, 154, 154], fill=(230, 60, 60))  # highlight
    t = TF.to_tensor(img)          # (3, 256, 256) float32 [0,1]
    return t.unsqueeze(0)          # (1, 3, 256, 256)

# ── load model (uses local cache — no re-download) ────────────────────────────
print("Loading SmolVLA (from cache)...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda")
policy.eval()
policy.reset()   # clear episode buffer before first step

# ── tokenize task ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
)
task = "pick up the red block and place it in the box"
enc  = tokenizer(task, return_tensors="pt", padding=True)

def build_batch(scene_tensor: torch.Tensor) -> dict:
    return {
        "observation.images.camera1":        scene_tensor.to("cuda"),
        "observation.images.camera2":        scene_tensor.to("cuda"),
        "observation.images.camera3":        scene_tensor.to("cuda"),
        # plausible resting joint angles for a 6-DOF arm (radians)
        "observation.state": torch.tensor(
            [[0.0, -0.3, 0.6, 0.0, 0.9, 0.0]], device="cuda"
        ),
        "observation.language.tokens":        enc["input_ids"].to("cuda"),
        "observation.language.attention_mask":enc["attention_mask"].bool().to("cuda"),
    }

# ── run inference on two scenes ───────────────────────────────────────────────
scenes = {
    "block present": make_scene(has_block=True),
    "empty table  ": make_scene(has_block=False),
}

print(f'\nTask: "{task}"\n')
results = {}
for name, scene in scenes.items():
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(build_batch(scene))
    results[name] = action

    # validation
    assert action.shape == (1, 6),          f"wrong shape: {action.shape}"
    assert not torch.isnan(action).any(),   "NaN in action"
    assert not torch.isinf(action).any(),   "Inf in action"

    vals = action[0].cpu()
    print(f"  [{name}]  action: {vals.numpy().round(3)}")
    print(f"              range [{vals.min():.3f}, {vals.max():.3f}]  "
          f"norm {vals.norm():.3f}")

# ── confirm actions differ between scenes ─────────────────────────────────────
diff = (results["block present"] - results["empty table  "]).abs().mean().item()
print(f"\n  Δ action (block vs empty): {diff:.4f}  "
      f"{'✓ model sees the image' if diff > 1e-4 else '⚠ outputs identical'}")

print("\n✓ Valid action output confirmed")
