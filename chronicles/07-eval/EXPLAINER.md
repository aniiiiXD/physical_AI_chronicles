# Chapter 07 — Evaluation and Profiling Explainer

This chapter has four files, each answering a different question:

| File | Question answered |
|------|-----------------|
| `simpler_eval.py` | Does SmolVLA succeed at a robot manipulation task? |
| `profile_nsight.sh` + `_forward_one.py` | What is the GPU doing during one forward pass? |
| `latency.py` | How fast does SmolVLA run, and which stage is the bottleneck? |

---

## `simpler_eval.py`

### What SimplerEnv is

**SimplerEnv** is a robotics benchmark built on MuJoCo (physics simulator)
and SAPIEN (renderer). It provides standardised environments for evaluating
robot policies:

- **WidowX** environments: 6-DOF tabletop arm tasks (pick-and-place, stacking)
- **Google Robot** environments: mobile manipulation (drawer opening, table
  wiping)
- Physics-based success detection: the environment auto-detects task completion
  by checking object poses, not human-defined rewards

We target `widowx_carrot_on_plate-v0` because WidowX is a 6-DOF arm — the
closest match to SmolVLA's 6-DOF action output.

### `MUJOCO_GL=egl`

```bash
MUJOCO_GL=egl python simpler_eval.py
```

MuJoCo's renderer needs a graphics context. On a machine with no monitor
(a server, an SSH session, or a headless GPU node), the default GLX backend
tries to connect to an X11 display and fails with:
```
EGLError: couldn't get EGL version
```
Setting `MUJOCO_GL=egl` switches to the EGL backend, which creates an
off-screen OpenGL context without needing a display. This is how all server-
side ML rendering works (cloud GPUs, CI machines, robot computers in
production).

### `SmolVLAAgent.act` (lines 68–91)

```python
def act(self, obs: dict) -> np.ndarray:
    raw = obs.get("image", obs.get("rgb", np.zeros((256,256,3), dtype=np.uint8)))
    img = TF.to_tensor(
        Image.fromarray(raw.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    ).unsqueeze(0).cuda()
```

**`obs.get("image", obs.get("rgb", ...))`**: SimplerEnv versions differ in
their observation key names. Some use `"image"`, others `"rgb"`. The nested
`.get` with a fallback zero-image handles both and degrades gracefully if
neither key exists.

**`.resize((256, 256))`**: SimplerEnv renders at its native resolution
(typically 480×640 or similar). SmolVLA was trained on 256×256 images.
PIL's `resize` is a CPU operation; this is part of the preprocessing cost
measured by `latency.py`.

```python
state_raw = obs.get("agent_qpos", obs.get("tcp_pose", np.zeros(6)))
state = torch.tensor(np.asarray(state_raw, dtype=np.float32)[:6]).unsqueeze(0).cuda()
```

**`agent_qpos`** is the joint position vector (angles in radians). We take
only the first 6 elements because SmolVLA expects a 6-DOF state. WidowX
may report additional values (gripper, timestamp). `[:6]` is a safe truncation.

### Action padding/trimming (lines 111–114)

```python
act_dim = env.action_space.shape[0]
if len(action) < act_dim:
    action = np.pad(action, (0, act_dim - len(action)))
else:
    action = action[:act_dim]
```

SmolVLA outputs 6-DOF joint positions. SimplerEnv's WidowX environments
may have a different action dimension (e.g., 7 if gripper is included).
We pad with zeros (gripper stays closed) or trim (if SmolVLA outputs more
than needed). This is a **domain gap** — a proper deployment would use an
inverse kinematics layer to convert SmolVLA's joint targets to the env's
action space.

### Why the success rate will be low

SmolVLA was trained on SO-100/SO-101 robot data from Hugging Face. SimplerEnv
uses WidowX kinematics. The visual appearance, camera viewpoints, joint
configurations, and action conventions are all different from training.

This is expected and educational: the evaluation loop is correct, the model
runs without errors, and you can measure how much the domain gap hurts.
A policy trained specifically on WidowX or fine-tuned for SimplerEnv would
show real performance.

---

## `_forward_one.py` and `profile_nsight.sh`

These two files work together: the shell script runs the Python script under
the `nsys` profiler.

### Why separate files?

If we put the profiled code directly in the shell script, `nsys` would
capture everything including Python interpreter startup, module imports,
and model loading. We want to profile only the actual inference, not warmup.

`_forward_one.py` handles its own warmup (2 passes), then opens the capture
window. The shell script handles the `nsys` invocation and result reporting.

### The profiler API boundary (lines 46–52 in `_forward_one.py`)

```python
policy.reset()
profiler.start()              # tell nsys: start recording NOW
with torch.no_grad():
    action = policy.select_action(batch)
torch.cuda.synchronize()      # wait for all GPU kernels to finish
profiler.stop()               # tell nsys: stop recording
```

**`torch.cuda.profiler.start/stop`** maps directly to `cudaProfilerStart()/Stop()`.
When the shell script passes `--capture-range=cudaProfilerApi` to nsys, the
profiler collects data only between these two calls.

Without this scoping: nsys would record all 2 warmup passes + model load +
tokenization. The timeline would be cluttered with JIT compilation and
one-time initialization noise, making it hard to identify per-inference
kernel patterns.

**`torch.cuda.synchronize()`**: CUDA kernels execute asynchronously. When
Python calls `policy.select_action(batch)`, it enqueues GPU work and returns
immediately — the GPU is still running. Without the synchronize, `profiler.stop()`
might fire before the last kernel finishes, truncating the capture.

### nsys flags explained (in `profile_nsight.sh`)

```bash
nsys profile \
    --output=smolvla_forward \
    --force-overwrite=true \
    --trace=cuda,cudnn,cublas,osrt \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --stats=true \
    python _forward_one.py
```

| Flag | Meaning |
|------|---------|
| `--output=smolvla_forward` | Write `smolvla_forward.nsys-rep` |
| `--force-overwrite=true` | Overwrite if the file already exists |
| `--trace=cuda,cudnn,cublas,osrt` | Capture GPU kernels, cuDNN ops, cuBLAS calls, and OS runtime (CPU thread API) |
| `--cuda-memory-usage=true` | Track GPU memory allocations over time |
| `--capture-range=cudaProfilerApi` | Only record between `cudaProfilerStart()` and `cudaProfilerStop()` |
| `--capture-range-end=stop` | Stop capture on `cudaProfilerStop()` (not on process exit) |
| `--stats=true` | Print a text summary after profiling |

### What to look for in the timeline

When you open `smolvla_forward.nsys-rep` in Nsight Systems GUI:

1. **Top CUDA kernels by duration**: the dominant kernel name tells you
   where time is spent. For SmolVLA you'll typically see attention kernels
   (`flash_fwd_kernel` or `cutlass_gemm`) and the diffusion denoising steps.

2. **Gaps between kernels**: white space on the GPU timeline means the GPU
   is idle, waiting for the CPU to launch the next kernel. This is called
   **CPU-GPU overlap inefficiency** — common in Python-driven inference because
   Python has overhead launching each operation.

3. **cudaMemcpy blocks**: host-to-device transfers for the input batch. If
   these are large, consider pre-pinning your input tensors.

4. **cuDNN vs cuBLAS time split**: cuDNN handles convolutions and attention;
   cuBLAS handles linear layers. The split tells you if the vision backbone
   (conv-heavy) or the transformer layers (matmul-heavy) dominate.

---

## `latency.py`

### CudaTimer (lines 35–43)

```python
class CudaTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self) -> float:
        self.e.record()
        torch.cuda.synchronize()
        return self.s.elapsed_time(self.e)
```

Identical in concept to `GpuTimer` in `vec_add.cu`. CUDA events are
inserted into the GPU command stream at `record()` time; `elapsed_time`
queries the hardware-measured interval between them.

`torch.cuda.synchronize()` in `stop()` blocks the CPU until the GPU
has processed the stop event. Without it, `elapsed_time` would be called
before the end event has fired, returning 0.

### PyTorch forward hooks for sub-stage timing (lines 73–88)

```python
def _hook_vlm_start(module, input):
    module._t = CudaTimer(); module._t.start()

def _hook_vlm_end(module, input, output):
    stage_times["vlm_encode"].append(module._t.stop())

vlm = policy.model.vlm_with_expert
vlm.register_forward_pre_hook(_hook_vlm_start)
vlm.register_forward_hook(_hook_vlm_end)
```

PyTorch modules support **forward hooks**: callbacks that fire before/after
`module.forward()`. This lets us time a specific sub-module without
modifying its source code.

- `register_forward_pre_hook(fn)`: `fn(module, input)` called just before forward
- `register_forward_hook(fn)`: `fn(module, input, output)` called just after

We attach to `vlm_with_expert` — the SmolVLM2 backbone. The VLM is one big
module, so start/stop around it captures the full vision-language encoding
stage. The diffusion denoiser is the remainder: `diffusion ≈ total − vlm_encode`.

**Why try/except (line 84)**:
```python
try:
    vlm = policy.model.vlm_with_expert
    ...
except AttributeError:
    pass
```
The internal attribute name `vlm_with_expert` depends on the exact LeRobot
version. If it changes, we fall back to total-only timing instead of crashing.

### Warmup reps (lines 91–96)

```python
for _ in range(WARMUP_REPS):
    policy.reset(); policy.select_action(batch)
torch.cuda.synchronize()
```

**Why warmup?**
1. **PyTorch JIT / `torch.compile`**: on first run, PyTorch traces the
   computation graph and compiles CUDA kernels. This can take seconds.
   Warmup reps absorb this cost so benchmark reps measure steady-state.
2. **cuDNN autotuning**: cuDNN benchmarks multiple convolution algorithms
   on first use and picks the fastest. This only happens once per config.
3. **GPU power state**: the GPU may start in a low-power state and ramp up
   after a few kernels.

Without warmup, the first measured rep would be 5–20× slower than
steady-state, severely biasing the mean.

### p95 latency (line 131)

```python
s = sorted(samples)
p95 = s[int(len(s) * 0.95)]
```

**Mean** tells you average throughput. **p95** (95th percentile) tells you
worst-case latency in a real run — 19 out of 20 calls are faster than this.
For real-time robot control, p95 matters: a single slow forward pass causes
a control hiccup that could destabilise the robot.

### Control Hz and action chunking (lines 152–165)

```python
hz = 1000.0 / total_p50  # ms → Hz
```

Control frequency = how many times per second the policy updates the robot's
joint targets. For context:
- Industrial robot arms run at 1000 Hz (position control loop)
- Most teleoperation demos run at 30–50 Hz
- SmolVLA with full diffusion denoising typically runs at 3–8 Hz on a 3060

**Action chunking** is SmolVLA's solution: one inference call predicts the
next 16 or 32 future joint positions. The robot executes them open-loop
(without re-running inference) and only calls the policy again after the
chunk is consumed. Effective control rate = Hz × chunk_size.

At 5 Hz inference with chunk_size=16, the effective planning rate is
80 Hz — fast enough for manipulation tasks. The tradeoff: the robot is
"committed" to 320ms of future motion before it can respond to a scene change.

---

## The full picture: what you now know

After Chapters 05–07 you have closed the loop from hardware to robot:

```
Chapter 05: CUDA kernels
    → GPU thread/block model
    → shared memory tiling
    → memory bandwidth vs compute throughput
    → why cuBLAS beats custom kernels (tensor cores, double-buffering)

Chapter 06: SmolVLA load and inference
    → VLA model architecture (vision encoder + diffusion action head)
    → action chunking
    → input format (3 cameras + state + tokenised instruction)
    → dtype pitfalls in mixed-precision inference

Chapter 07: Evaluation and profiling
    → SimplerEnv for standardised robot policy evaluation
    → nsys for GPU timeline profiling (kernel-level visibility)
    → CUDA event timing for component-level latency
    → control Hz and its relationship to action chunking
```

The common thread: **memory is the bottleneck**. In CUDA kernels it's DRAM
latency. In SmolVLA inference it's moving 500M parameters through the GPU's
execution units fast enough to keep the robot moving. Every optimization —
tiling, tensor cores, quantisation, action chunking — is ultimately about
getting the right data to the right compute unit at the right time.
