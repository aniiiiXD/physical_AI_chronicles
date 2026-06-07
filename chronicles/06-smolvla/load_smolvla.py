"""
load_smolvla.py — load SmolVLA checkpoint and run a smoke-test inference

Setup:
    cd ~/lerobot
    source ~/lerobot-env/bin/activate
    huggingface-cli login          # needs HF token with read access
    python load_smolvla.py

SmolVLA architecture:
    SmolVLM-500M vision-language backbone
        + action head (predicts robot joint positions)
    Input:  camera image(s) + robot joint state + language instruction
    Output: action chunk (next N joint positions)
    VRAM:   ~2.5 GB in bfloat16 — fits in RTX 3060 6GB
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ── load checkpoint ───────────────────────────────────────────────────────────
print("Loading SmolVLA from HuggingFace Hub...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda", dtype=torch.bfloat16)
policy.eval()

print(f"\nLoaded on : {next(policy.parameters()).device}")
print(f"VRAM used : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")

# ── smoke-test inference ──────────────────────────────────────────────────────
# Dummy inputs — replace with real camera frames and robot state later.
# image shape: (batch, channels, height, width)
# state shape: (batch, dof) — 14-DOF for common bimanual robot setups
image = torch.zeros(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
state = torch.zeros(1, 14,       device="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    action = policy.select_action({
        "observation.images.top": image,
        "observation.state":      state,
    })

print(f"\nAction shape : {action.shape}")
print(f"Action dtype : {action.dtype}")
print(f"Action sample: {action[0, :4]}  ...")
print("\n✓ SmolVLA inference working")
