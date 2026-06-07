# SmolVLA — Load and Inference Explainer

This chapter has two files: `load_smolvla.py` loads the model and prints
its input/output structure; `inference.py` runs it on meaningful synthetic
scenes and validates the output.

---

## What SmolVLA is (big picture)

SmolVLA is a **Vision-Language-Action** model — a robot policy that maps:

```
[camera images] + [language instruction] + [robot joint state]
        ↓
[sequence of future joint positions]
```

It's a 500M-parameter model with two stages chained together:

1. **SmolVLM2-500M** (backbone): a vision-language transformer. Takes images
   and text tokens → produces contextual embeddings. Think of it as
   "understanding what you see and what you're asked to do."

2. **Diffusion action head**: takes those embeddings + current robot state →
   iteratively denoises random Gaussian noise into a clean action trajectory.
   Think of it as "translate understanding into motion."

The word "small" is relative — 500M parameters is tiny compared to GPT-4 but
designed to run on-device on a robot's embedded GPU.

---

## `load_smolvla.py`

### Imports

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
```

**LeRobot** is Hugging Face's robotics library. Its module path changed in
version 0.5.2 from `lerobot.common.policies.smolvla` to
`lerobot.policies.smolvla`. If you see `ModuleNotFoundError`, this is why.

**`transformers`** is Hugging Face's model library. SmolVLA reuses the
SmolVLM2 tokenizer from it — the same tokenizer used for the language
backbone.

### Loading the model

```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda")   # float32 — important
policy.eval()
```

**`from_pretrained`** downloads the model weights from Hugging Face Hub
on first run (~1 GB), caches them at `~/.cache/huggingface/`, and loads
them. Subsequent runs are instant (from cache).

**Why `float32` and not `bfloat16`?**
The diffusion denoiser creates noise tensors using `torch.randn()`, which
defaults to float32. If the model is in bfloat16, the denoiser's internal
`torch.where(condition, tensor_bf16, noise_f32)` call hits a dtype mismatch:
```
RuntimeError: mat1 and mat2 must have same dtype, Float and BFloat16
```
Loading as float32 avoids this. A proper fix would use
`policy.to(torch.bfloat16)` AND set the denoising noise dtype — left
as a future exercise.

### Reading `policy.config.input_features`

```python
for name, feat in policy.config.input_features.items():
    print(f"  {name}: {feat}")
```

This is how you discover what the model actually expects. SmolVLA has three
camera slots named `camera1`, `camera2`, `camera3` — not `top`, `wrist`, or
any other convention. If you guess the key names wrong you get:
```
ValueError: All image features are missing from the observation batch.
```
Always read the config first.

### Tokenising the task instruction

```python
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
)
enc = tokenizer(task, return_tensors="pt", padding=True)
```

SmolVLA expects the language instruction as token IDs (integers), not a raw
string. The tokenizer converts text → integer sequence using the SmolVLM2
vocabulary. `return_tensors="pt"` returns PyTorch tensors. `padding=True`
pads to the longest sequence in the batch (only matters if you're batching
multiple tasks simultaneously).

---

## `inference.py`

### `make_scene` (lines 21–33)

```python
def make_scene(has_block: bool = True) -> torch.Tensor:
    img = Image.new("RGB", (256, 256), color=(180, 180, 175))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 200, 256, 256], fill=(100, 80, 60))
    if has_block:
        draw.rectangle([100, 100, 156, 156], fill=(210, 45, 45))
        draw.rectangle([102, 102, 154, 154], fill=(230, 60, 60))
    return TF.to_tensor(img).unsqueeze(0)
```

We need two visually different scenes to test whether the model responds to
its input (if both scenes produce identical actions, something is wrong —
the model might be ignoring the images entirely).

- **Grey rectangle** (`180, 180, 175`): plain workspace background
- **Dark edge** (`100, 80, 60`): depth cue — makes the scene look like a
  table with a near edge
- **Red block** (`210, 45, 45` + lighter highlight): the manipulation target

`TF.to_tensor` converts a PIL Image (uint8 pixels 0–255) to a float32
tensor with values in [0, 1], with shape `(3, 256, 256)`. `.unsqueeze(0)`
adds a batch dimension → `(1, 3, 256, 256)`.

SmolVLA expects `(batch, channels, height, width)` — the standard PyTorch
image format.

### `build_batch` (lines 49–60)

```python
def build_batch(scene_tensor):
    return {
        "observation.images.camera1": scene_tensor.to("cuda"),
        "observation.images.camera2": scene_tensor.to("cuda"),
        "observation.images.camera3": scene_tensor.to("cuda"),
        "observation.state": torch.tensor([[0.0, -0.3, 0.6, 0.0, 0.9, 0.0]], device="cuda"),
        "observation.language.tokens": enc["input_ids"].to("cuda"),
        "observation.language.attention_mask": enc["attention_mask"].bool().to("cuda"),
    }
```

SmolVLA takes a dictionary of named observations. The keys must match
exactly what `policy.config.input_features` specifies.

**Three camera slots with the same image**: SmolVLA was trained with three
cameras (typically overhead, wrist, side). We only have one synthetic image,
so we pass it for all three. This is fine for testing — the model still
produces a valid action, just not one conditioned on three distinct views.

**Robot state** `[0.0, -0.3, 0.6, 0.0, 0.9, 0.0]`: plausible resting
configuration for a 6-DOF arm in radians. Joint angles near these values
mean the arm is roughly upright, not folded.

**`.bool()` on attention_mask**: the tokenizer produces a `LongTensor`
(integer 0/1). SmolVLA uses the mask in `torch.where(mask, ...)` which
requires a boolean tensor. Passing Long gives:
```
RuntimeError: where expected condition to be boolean tensor, but got Long
```

### `policy.reset()` before each episode (lines 72–73)

```python
policy.reset()
action = policy.select_action(build_batch(scene))
```

SmolVLA uses **action chunking**: on the first call it computes a chunk of
N future actions. Subsequent calls return the pre-computed actions without
running inference. `reset()` clears this buffer so the next `select_action`
call performs a fresh forward pass.

If you forget `reset()` between scenes, the second scene's action comes
from the first scene's chunk — the model hasn't actually "seen" the new
image.

### Output validation (lines 77–84)

```python
assert action.shape == (1, 6)
assert not torch.isnan(action).any()
assert not torch.isinf(action).any()
```

Shape `(1, 6)`: batch size 1, 6-DOF joint position output.
NaN check: if the model has a dtype bug or the diffusion denoiser diverges,
NaN propagates silently through the network and appears here.
Inf check: overflow in fp32 arithmetic.

### The Δ check (lines 87–88)

```python
diff = (results["block present"] - results["empty table  "]).abs().mean()
```

If `diff < 1e-4`, both scenes produced the same action — the model is
ignoring the image input (or producing a constant output regardless of
observation). A healthy diff around 0.1–0.5 confirms the visual backbone
is actually processing the scene. Our benchmark got `Δ = 0.1261`.

---

## The error trail — what went wrong during setup

Getting SmolVLA running took 7 sequential fixes. Each one was caused by
a different layer of the system. Here they are in order:

| Error | Root cause | Fix |
|-------|-----------|-----|
| `ModuleNotFoundError: lerobot.common.policies` | LeRobot 0.5.2 restructured its module paths | Change import to `lerobot.policies.smolvla` |
| `ImportError: 'transformers' is required` | transformers not in the environment | `pip install transformers` |
| `ValueError: All image features are missing` | Used wrong camera key name (`top` instead of `camera1`) | Read `policy.config.input_features` |
| `KeyError: 'observation.language.tokens'` | Model requires tokenised text; none provided | Add `enc["input_ids"]` to batch |
| `KeyError: 'observation.language.attention_mask'` | Attention mask also required | Add `enc["attention_mask"]` to batch |
| `RuntimeError: expected boolean tensor, got Long` | Mask was integer type | Cast to `.bool()` |
| `RuntimeError: mat1 and mat2 must have same dtype, Float and BFloat16` | Denoiser creates float32 noise; model was in bfloat16 | Load model in float32 (default, no `.to(bfloat16)`) |

The pattern: SmolVLA has strict input requirements (three cameras + state +
tokenised instruction + mask, all correct dtypes) because it was designed
for a real training pipeline where all inputs are always present. Loading it
outside that pipeline means manually satisfying every requirement.

---

## How SmolVLA's diffusion action head works

The action head is a **DDPM** (Denoising Diffusion Probabilistic Model)
applied to robot actions instead of images.

1. Start with pure Gaussian noise `x_T ~ N(0, I)` in the action space
2. Run a denoising loop for T steps (typically 100 during training, fewer
   during inference):
   ```
   x_{t-1} = denoise(x_t, t, conditioning)
   ```
   where `conditioning` is the embedding from SmolVLM2
3. After T steps, `x_0` is the predicted action trajectory

Each step `denoise(...)` is a small transformer that takes the noisy action,
the timestep embedding, and the visual-language conditioning, and predicts
the noise to subtract. After many steps the noise cancels and you're left
with a clean action.

**Why diffusion for actions?** Robot manipulation has multi-modal action
distributions — given the same observation, multiple different trajectories
might all succeed. A direct regression model would predict the average of
these modes (a useless middle path). Diffusion can represent the full
distribution and sample from it.

---

## What SmolVLA cannot do (and why we test it anyway)

SmolVLA was trained on LeRobot datasets from Hugging Face — teleoperation
demonstrations with specific robots (SO-100 / SO-101 arms). Our synthetic
PIL images are not from its training distribution, and the real test
environments in Chapter 07 (SimplerEnv) use different robot kinematics.

The inference test proves: the **pipeline is correct and the model runs**.
A domain-matched checkpoint on a real robot would produce behaviours that
actually work.
