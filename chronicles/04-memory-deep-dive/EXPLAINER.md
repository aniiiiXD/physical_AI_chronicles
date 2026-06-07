# Memory Deep-Dive — Interactive Page Explainer

`index.html` is a self-contained educational page on CPU and GPU memory
architecture. Everything — styles, scripts, animations — lives in one file
so you can open it in any browser without a server.

---

## What it teaches (top to bottom)

The page builds up the same core idea in six sections:

1. **Hero / bit-rain** — sets the mood; visually establishes "data in motion"
2. **CPU vs GPU side-by-side** — the fundamental architecture difference
3. **Memory hierarchy** — registers → L1 → L2 → L3 → DRAM, with latencies
4. **Warp execution stepper** — interactive: step through SIMT divergence
5. **Occupancy calculator** — live: change block size, see active warps
6. **Memory access tracer** — animated: coalesced vs strided vs random

---

## Why one HTML file?

A single `.html` file with inline CSS and JS is the simplest possible
distribution format for a reference document you'll open locally. No npm
install, no bundler, no server. The tradeoff is a long file — offset by the
fact that you never have to track asset paths.

---

## The CSS architecture

### CSS variables (`:root`)

```css
:root {
  --bg:      #07090F;
  --cpu:     #3D9EFF;
  --gpu:     #00FFAA;
}
```

All colours are defined once as custom properties. Every rule that needs
blue uses `var(--cpu)` — change the variable, change the whole page. This
is the CSS equivalent of named constants in code.

**Why two accent colours?** Blue (`--cpu`) and green (`--gpu`) are
perceptually distinct under deuteranopia (the most common colour vision
deficiency). They also have immediate association: blue = cool/established
(CPUs have been around since the 70s), green = new/fast (GPU compute is
relatively recent).

### `--cpu-dim` / `--gpu-dim` / `--cpu-mid` / `--gpu-mid`

Each colour has three tints:
- `dim` (~10-12% opacity) — background fills, hover states
- `mid` (~22-25% opacity) — borders, selected states, strong fills
- full — text, icons, key labels

This gives four visual weights without adding new colours.

### Surface layers (`--sf1`, `--sf2`, `--sf3`)

```css
--sf1: #0C1018;
--sf2: #121924;
--sf3: #192130;
```

Three shades of near-black create depth. Cards sit on `--sf2`; their
headers use `--sf3`; the page background is `--bg` (#07090F, the darkest).
No box shadows needed — elevation comes from colour alone.

---

## The bit-rain hero

The animated rain uses an HTML `<canvas>` element drawn by JavaScript.

### How it works

```js
const cols = Math.floor(canvas.width / 14);
const drops = new Array(cols).fill(0);

function draw() {
    ctx.fillStyle = 'rgba(7, 9, 15, 0.05)';  // near-transparent overlay
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // ... draw each column's current character
    drops[i]++;
    if (drops[i] * 14 > canvas.height && Math.random() > 0.975)
        drops[i] = 0;   // reset to top with ~2.5% chance per frame
}
setInterval(draw, 50);  // ~20 fps
```

**The fading trail trick**: instead of clearing the canvas every frame
(which would show only the current character per column), we draw a
semi-transparent black rectangle over the entire canvas. Characters from
previous frames are partially obscured but not erased, creating the fade.
The opacity `0.05` controls fade speed — lower = longer trails.

**Column reset**: each column resets to the top only when it reaches the
bottom AND a random check passes (97.5% chance to continue). This makes
columns desynchronised and organic-looking.

---

## CPU vs GPU comparison section

### The core architectural difference

```
CPU (8-core):                    GPU (3060: 28 SM × 128 cores):
┌──┐ ┌──┐ ┌──┐ ┌──┐             ┌──┬──┬──┬──┬──┬──┬──┬──┐
│C1│ │C2│ │C3│ │C4│             │  │  │  │  │  │  │  │  │ × 28 SMs
└──┘ └──┘ └──┘ └──┘             └──┴──┴──┴──┴──┴──┴──┴──┘
 Large cache, out-of-order        Small cache, SIMT,
 execution, branch prediction     3584 shaders total
```

The design philosophies are opposite:
- **CPU**: minimise latency for sequential work. Big caches and out-of-order
  execution hide memory latency for a handful of threads.
- **GPU**: maximise throughput for parallel work. Instead of hiding latency,
  hide it by switching to another warp. Thousands of threads in flight at
  once means there's always something to run while waiting for memory.

---

## Memory hierarchy section

Each level is shown with its typical latency for an RTX 3060:

| Level | Size | Latency | Notes |
|-------|------|---------|-------|
| Registers | 64 KB/SM | ~1 cycle | Per-thread; fastest possible |
| L1 / Shared | 128 KB/SM | ~20 cycles | Shared within a block |
| L2 cache | 3 MB | ~200 cycles | Shared across all SMs |
| GDDR6 (VRAM) | 6 GB | ~600 cycles | Main video memory |
| PCIe / system RAM | 16+ GB | ~10,000 cycles | Crossing the PCIe bus |

The key insight: **register and shared memory are not just faster, they are
programmer-controlled**. When you write `__shared__ float tile[32][32]` in
a CUDA kernel, you are explicitly telling the hardware "keep this in the
fast scratchpad, not DRAM." The cache hierarchy on a CPU is automatic — the
hardware decides what to keep. On a GPU, shared memory is manual.

---

## Warp execution stepper

### What a warp is

The GPU executes threads in groups of 32 called **warps**. All 32 threads
in a warp execute the same instruction at the same time (SIMT — Single
Instruction, Multiple Threads). This is like having 32 calculators wired to
the same instruction decoder: they all do the same operation, but each on
its own data.

### Warp divergence

```c
if (threadIdx.x < 16) {
    do_path_A();   // threads 0–15 execute this
} else {
    do_path_B();   // threads 16–31 execute this
}
```

When threads in the same warp take different branches, the GPU runs both
paths sequentially with a mask — threads that "didn't take this path" are
disabled (their writes become no-ops) while the other half runs. This is
**warp divergence**: a 32-thread warp doing the work of 16, then 16 again.
Peak throughput is halved.

The stepper in the page lets you click through this: you see threads 0–15
lit up for path A, then 16–31 lit up for path B, with the inactive threads
greyed out.

**Why does this matter for robotics AI?** Diffusion models (like SmolVLA's
action head) do denoising loops that branch based on step count. Keeping
warps coherent in these loops is a real optimization target.

### `__syncthreads()`

This instruction appears in the tiled matmul kernel:

```c
__syncthreads();  // wait until all threads in the block have loaded their tile
for (int k ...) acc += As[ty][k] * Bs[k][tx];
__syncthreads();  // wait before overwriting the tile
```

It is a barrier — no thread can proceed past it until every thread in the
block has reached it. Without the first one, some threads would start reading
`As[ty][k]` before other threads have written their values into it — a race
condition. Without the second one, fast threads would start overwriting the
tile before slow threads have finished using it.

---

## Occupancy calculator

**Occupancy** = (active warps on an SM) / (maximum warps the SM can hold).

The RTX 3060's SM can hold at most 48 warps simultaneously. If your kernel
only schedules 16, occupancy is 33% — 2/3 of the SM sits idle.

### What limits occupancy?

1. **Register pressure**: each thread uses some registers. If your kernel
   uses 64 registers/thread with 256 threads/block, that's 16,384 registers
   per block. An SM has 65,536 registers total, so only 4 blocks can be
   resident at once = 4 × 8 warps = 32 warps = 67% occupancy.

2. **Shared memory**: if your block allocates 32 KB of shared memory and the
   SM has 64 KB total, only 2 blocks can be resident = 2 × 8 warps = 16 = 33%.

3. **Block size**: if each block has 32 threads = 1 warp, and the SM holds
   max 32 blocks, you get 32 warps = 67%. With 256 threads = 8 warps/block
   and max 32 blocks → 48 warps = 100%.

The calculator lets you slide the block size and see how these limits
interact on the specific SM of an RTX 3060 (sm_86).

---

## Memory access tracer

Three patterns visualised for a 1D array:

**Coalesced** (best):
```
Thread:   0  1  2  3  4  5  6  7 ...
Address:  0  1  2  3  4  5  6  7 ...
```
All 32 threads in a warp access consecutive addresses → the GPU hardware
merges these into a single 128-byte transaction. Efficiency: 100%.

**Strided** (2× stride):
```
Thread:   0  1  2  3  4  5 ...
Address:  0  2  4  6  8  10 ...
```
Addresses are still sorted but non-consecutive. The hardware still merges
what it can, but you're accessing 256 bytes of address space to read 128
bytes of data. Efficiency: 50%.

**Random / scattered** (worst):
```
Thread:   0   1   2   3 ...
Address:  57  3   891  14 ...
```
Each thread's address falls in a different cache line. 32 separate 128-byte
transactions to read 4 bytes each. Efficiency: ~3%. This is why pointer
chasing (Chapter 01) is so slow: each pointer read is random.

**The connection to our benchmarks**: vec_add has perfectly coalesced access
(thread i reads element i). Tiled matmul's shared memory reads are also
coalesced within each warp. The naive matmul reading B column-major is
strided — which is one reason it's slow.

---

## Fonts

- **Syne** (display): geometric, slightly futuristic — evokes circuit diagrams
- **IBM Plex Mono** (code): technical, legible, intentionally "machine"
- **DM Sans** (body): neutral and readable for long prose sections

These are loaded from Google Fonts via `<link rel="preconnect">` — the
preconnect hint tells the browser to open a TCP connection to fonts.googleapis.com
before it actually needs any fonts, cutting latency.

---

## How to run it

```bash
open chronicles/04-memory-deep-dive/index.html   # macOS
xdg-open chronicles/04-memory-deep-dive/index.html  # Linux
```

Or just drag the file into any browser tab. No server needed.
