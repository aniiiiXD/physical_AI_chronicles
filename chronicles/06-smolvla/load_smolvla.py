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
from transformers import AutoTokenizer
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
# Read expected input features directly from the loaded config so the
# dummy batch always matches, regardless of checkpoint variant.
print("\nExpected inputs:")
state_dim = None
image_keys = {}
for key, feat in policy.config.input_features.items():
    print(f"  {key}: {feat.type.value}  shape={feat.shape}")
    if feat.type.value == "VISUAL":
        image_keys[key] = feat.shape
    else:
        state_dim = feat.shape[0]

# tokenize the language instruction — SmolVLA requires text input
# (it's a vision-LANGUAGE-action model, text tells it what task to do)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
task = "pick up the red block and place it in the box"
encoded = tokenizer(task, return_tensors="pt", padding=True)
lang_tokens = encoded["input_ids"].to("cuda")
lang_mask   = encoded["attention_mask"].to("cuda")
print(f"\nTask: \"{task}\"")
print(f"Token shape: {lang_tokens.shape}")

# build dummy batch from config
batch = {}
for key, shape in image_keys.items():
    batch[key] = torch.zeros(1, *shape, device="cuda", dtype=torch.bfloat16)
batch["observation.state"]                   = torch.zeros(1, state_dim, device="cuda", dtype=torch.bfloat16)
batch["observation.language.tokens"]         = lang_tokens
batch["observation.language.attention_mask"] = lang_mask

with torch.no_grad():
    action = policy.select_action(batch)

print(f"\nAction shape : {action.shape}")
print(f"Action dtype : {action.dtype}")
print(f"Action sample: {action[0, :4]}  ...")
print("\n✓ SmolVLA inference working")
