# Cache Benchmark — Line by Line Explainer

This document explains every line of `cache_benchmark.cpp`, the `Makefile`,
and introduces CMake.

---

## What the program is actually doing (big picture first)

Your CPU has multiple layers of memory, each faster but smaller than the next:

```
Registers  →  L1 cache (~32 KB, ~1 ns)
           →  L2 cache (~256 KB, ~3 ns)
           →  L3 cache (~8–32 MB, ~10 ns)
           →  RAM      (GBs, ~50–100 ns)
```

The benchmark creates an array of a specific size, then reads it in a
**random jumping pattern** (pointer chase). If the whole array fits in L1,
every read is fast (~1 ns). If it doesn't fit, data must be fetched from a
slower layer, and latency jumps. By repeating this for arrays of increasing
size, we can see exactly where each cache boundary is.

---

## The includes (lines 1–8)

```cpp
#include <iostream>    // std::cout — printing to terminal
#include <iomanip>     // std::setw, std::fixed, std::setprecision — table formatting
#include <vector>      // std::vector — a resizable array
#include <numeric>     // std::iota — fills a range with 0,1,2,3,...
#include <algorithm>   // std::swap — swaps two values
#include <cstring>     // std::memset — fills a block of memory with a value
#include <cstdint>     // uint64_t — an integer that is exactly 64 bits wide
#include <chrono>      // high-resolution timer for measuring time
```

C++ splits its standard library into many header files. You only pay for what
you include. Each `#include` pastes the declarations from that header into
your file so you can use those functions and types.

---

## The constants (lines 10–12)

```cpp
static constexpr size_t CACHE_LINE  = 64;
static constexpr size_t MAX_BYTES   = 256ULL * 1024 * 1024;
static constexpr size_t ACCESSES    = 1 << 26;
```

**`constexpr`** means "evaluate this at compile time, not at runtime." The
compiler substitutes the value everywhere the name appears — no memory is
allocated, no runtime cost.

**`static`** here means "this name is only visible inside this translation
unit (this .cpp file)." It prevents name collisions if you later link
multiple files together.

**`size_t`** is the type the standard uses for sizes and counts. On a 64-bit
machine it is a 64-bit unsigned integer.

- `CACHE_LINE = 64`: Almost every modern CPU loads memory in 64-byte chunks
  called cache lines. Accessing one byte drags the whole 64-byte line into
  cache. We stride by this amount to ensure each access touches a new line.

- `MAX_BYTES = 256ULL * 1024 * 1024`: 256 MB. The `ULL` suffix makes the
  literal an `unsigned long long` so the multiplication doesn't overflow a
  32-bit int before being stored. `1024 * 1024` = 1 MB, so `256 * 1024 *
  1024` = 256 MB.

- `ACCESSES = 1 << 26`: Bit-shift left by 26 = 2^26 = 67,108,864 (~67M).
  We do this many pointer chases per measurement so the timer is averaged
  over enough iterations to be statistically meaningful.

---

## The `measure` function (lines 14–46)

This is the heart of the benchmark. It takes the buffer, a size to use, and
a stride, and returns the average nanoseconds per memory access.

### Line 15: divide the buffer into slots

```cpp
size_t n = buf_size / stride;
```

If the buffer is 64 KB and the stride is 64 bytes, we have 1024 slots.
Each slot will hold one pointer — this is how we build the linked list.

### Lines 17–18: create an index list

```cpp
std::vector<size_t> idx(n);
std::iota(idx.begin(), idx.end(), 0);
```

`std::vector<size_t> idx(n)` creates a vector of `n` elements, all zero.
`std::iota` fills it with 0, 1, 2, … n-1. So `idx` is just [0, 1, 2, 3, …].
This is the ordered list of slot numbers we're about to shuffle.

### Lines 21–25: Fisher-Yates shuffle

```cpp
uint64_t rng = 0xdeadbeefcafe1234ULL;
for (size_t i = n - 1; i > 0; --i) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    std::swap(idx[i], idx[rng % (i + 1)]);
}
```

**Why shuffle?** Modern CPUs have a hardware prefetcher — a circuit that
watches your memory access patterns and pre-loads data it predicts you'll
need next. If you walk through memory sequentially (0, 64, 128, 192…) the
prefetcher will guess perfectly every time and fill the cache before you
even ask, making every access look like an L1 hit regardless of array size.
Randomising the order defeats it.

**The shuffle algorithm** (Fisher-Yates): starting from the last element,
swap it with a randomly chosen earlier element. This produces a uniformly
random permutation.

**The random number generator** is an xorshift64. Three XOR-shift operations
(`^=` with `<<` and `>>`) scramble the bits cheaply — no division, no
library calls, reproducible (same seed = same sequence every run).
`0xdeadbeefcafe1234` is just an arbitrary non-zero starting seed.

`rng % (i + 1)` gives a random number in [0, i], picking the swap target.

### Lines 28–29: embed the pointer chain

```cpp
for (size_t i = 0; i < n; ++i)
    *reinterpret_cast<size_t*>(buf + idx[i] * stride) = idx[(i + 1) % n] * stride;
```

This turns the buffer into a **linked list baked into raw memory**.

- `buf + idx[i] * stride`: address of slot `idx[i]` in the buffer.
- `reinterpret_cast<size_t*>(...)`: tells the compiler "treat whatever is at
  this address as a `size_t*` (a pointer to an integer)." `reinterpret_cast`
  is C++'s way of saying "I know what I'm doing, reinterpret these bytes."
- `= idx[(i+1) % n] * stride`: store the byte offset of the next slot in the
  shuffled sequence. The `% n` wraps the last element back to the first,
  forming a cycle.

After this loop, every slot contains the byte offset of the next slot to
visit — a classic singly-linked list, except the "pointers" are offsets into
`buf`.

### Lines 32–35: warm-up

```cpp
size_t pos = 0;
for (int w = 0; w < 3; ++w)
    for (size_t i = 0; i < n; ++i)
        pos = *reinterpret_cast<size_t*>(buf + pos);
```

We chase the chain three times before timing. This ensures:
1. The OS has actually mapped the pages into physical memory (first touch
   can be slow due to page faults).
2. The TLB (Translation Lookaside Buffer — a cache for virtual-to-physical
   address translations) is warmed up.
3. If the array fits in cache, it's already there when we start timing.

### Lines 37–40: the timed measurement

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
for (size_t i = 0; i < ACCESSES; ++i)
    pos = *reinterpret_cast<size_t*>(buf + pos);
auto t1 = std::chrono::high_resolution_clock::now();
```

`std::chrono::high_resolution_clock::now()` reads the CPU's nanosecond timer.
`auto` lets the compiler infer the type (it's a `time_point`, which is
verbose to write out). We bracket the chase loop with two snapshots and
subtract them to get elapsed time.

### Line 42: defeating dead-code elimination

```cpp
if (pos == static_cast<size_t>(-1)) std::cout << "";
```

The compiler is smart. If `pos` is never used after the loop, it might
realise the entire loop has no visible effect and delete it — giving us a
measured time of 0 ns. Using `pos` in a condition that can never be true
(no real `pos` will ever equal `SIZE_MAX`) forces the compiler to keep the
loop. `static_cast<size_t>(-1)` is the idiomatic way to get the maximum
value of `size_t` (wrapping -1 to all-ones).

### Lines 44–45: compute and return latency

```cpp
double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
return ns / static_cast<double>(ACCESSES);
```

`t1 - t0` gives a `duration`. Converting it to `duration<double, std::nano>`
expresses it as a floating-point number of nanoseconds. `.count()` extracts
the raw double. Dividing by ACCESSES gives the average cost per single access.

---

## `fmt_size` helper (lines 48–52)

```cpp
static std::string fmt_size(size_t bytes) {
    if (bytes >= 1024 * 1024)
        return std::to_string(bytes >> 20) + " MB";
    return std::to_string(bytes >> 10) + " KB";
}
```

`bytes >> 20` is a bit-shift right by 20, equivalent to dividing by 2^20 =
1,048,576 (1 MB). `bytes >> 10` divides by 1024 (1 KB). This converts raw
byte counts to human-readable strings like "32 MB" or "256 KB".

---

## `main` (lines 54–94)

### Line 55–56: allocate and initialise the buffer

```cpp
auto buf = std::make_unique<char[]>(MAX_BYTES);
std::memset(buf.get(), 1, MAX_BYTES);
```

`std::make_unique<char[]>(MAX_BYTES)` allocates 256 MB on the heap and wraps
it in a `unique_ptr` — a smart pointer that automatically frees the memory
when `buf` goes out of scope. You never need to call `delete`.

`std::memset(buf.get(), 1, MAX_BYTES)` fills every byte with 1. This forces
the OS to actually back the virtual allocation with physical RAM pages (lazy
allocation means pages don't exist until first write).

### Lines 58–74: the size sweep table

```cpp
static const size_t sizes[] = { 8*1024, 16*1024, … 128*1024*1024 };
```

An array of array sizes to test, going from 8 KB to 128 MB. `static const`
inside a function means the array is initialised once and lives for the
program's lifetime (not on the stack, which has limited space).

### Lines 76–80: print the table header

```cpp
std::cout << std::left
          << std::setw(12) << "Array size"
          << std::setw(12) << "Stride"
          << "Latency (ns/access)\n"
          << std::string(44, '-') << '\n';
```

`std::left` left-aligns the following fields.
`std::setw(12)` sets the next field's minimum width to 12 characters (padded
with spaces). `std::string(44, '-')` constructs a string of 44 dashes — the
separator line.

### Lines 82–93: the measurement loop

```cpp
for (size_t arr : sizes) {
    size_t stride = CACHE_LINE;
    if (arr / stride < 64) stride = arr / 64;
    ...
}
```

Range-based for loop — `arr` takes each value from `sizes` in turn.

The guard `if (arr / stride < 64) stride = arr / 64` ensures we always have
at least 64 slots, even for the smallest arrays. Without this, a tiny array
with a 64-byte stride might only have a handful of slots, making the
measurement noisy.

`buf.get()` returns the raw pointer from the `unique_ptr` — needed because
`measure` expects a plain `char*`.

`std::cout.flush()` forces the output to appear immediately rather than
being buffered. Useful so you see results as they're computed (large arrays
take time).

---

## The Makefile

```makefile
CXX      = clang++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
```

Variables. `CXX` is the conventional name for the C++ compiler. `CXXFLAGS`
are the flags passed to it:
- `-O2`: optimisation level 2 — the compiler reorganises your code for speed
  while keeping it correct.
- `-std=c++17`: use the C++17 standard (needed for `std::make_unique` with
  arrays and structured bindings).
- `-Wall -Wextra`: enable most warning messages — the compiler tells you
  about potential bugs.

```makefile
cache_benchmark: cache_benchmark.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
```

A **rule**: `cache_benchmark` depends on `cache_benchmark.cpp`. If the .cpp
is newer than the binary, run the recipe.
- `$@` expands to the target name (`cache_benchmark`).
- `$<` expands to the first dependency (`cache_benchmark.cpp`).
- `-o $@` means "output file named $@."

```makefile
.PHONY: run clean
```

Tells make that `run` and `clean` are not real files — just names for
commands. Without this, if a file named `clean` existed, `make clean` would
do nothing.

---

# What is CMake?

A `Makefile` is fine for one file, but as projects grow it becomes painful:
- You have to manually list every source file.
- It's hard to handle different operating systems (Windows uses different
  commands and file paths).
- It doesn't know about header dependencies automatically.

**CMake** solves this. It's a **build system generator** — you write a
`CMakeLists.txt` describing *what* to build, and CMake generates the actual
build files for your platform (`Makefile` on Linux/Mac, Visual Studio project
on Windows, Ninja files if you prefer Ninja, etc.).

```
You write:   CMakeLists.txt
CMake reads it and writes:  Makefile  (or .sln, or build.ninja, etc.)
You then run: make  (or ninja, or msbuild)
```

Think of CMake as the "meta build system" and make/ninja as the "actual
build system."

## Why use CMake over a plain Makefile?

| | Makefile | CMake |
|---|---|---|
| Cross-platform | No | Yes |
| Auto header deps | No (manual) | Yes |
| IDE integration | Poor | Excellent |
| Package finding | Manual | `find_package()` built-in |
| Multi-file projects | Verbose | Concise |

For a single-file toy benchmark a Makefile is fine. For any real project,
CMake is the standard.

---

## How to use CMake — step by step

### 1. The `CMakeLists.txt` file (see the one in this folder)

```cmake
cmake_minimum_required(VERSION 3.15)
project(cache_benchmark)
...
add_executable(cache_benchmark cache_benchmark.cpp)
```

This is the only file you need to write. CMake reads it.

### 2. Out-of-source build (the standard workflow)

Always build in a separate directory so generated files don't pollute your
source tree:

```bash
mkdir build        # create a build directory
cd build
cmake ..           # run CMake; reads ../CMakeLists.txt; generates Makefile here
make               # compile using the generated Makefile
./cache_benchmark  # run
```

Or in one line from the project root:

```bash
cmake -S . -B build && cmake --build build && ./build/cache_benchmark
```

- `-S .`  — source directory is here (where CMakeLists.txt lives)
- `-B build` — put generated files in ./build/
- `cmake --build build` — compile (equivalent to `cd build && make`)

### 3. Changing build type (Debug vs Release)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

`-DCMAKE_BUILD_TYPE=Release` passes the variable `CMAKE_BUILD_TYPE=Release`
to CMake, which enables `-O3` optimisations. `Debug` adds `-g` debug symbols.

### 4. Useful CMake commands

```bash
cmake --build build --clean-first   # clean then rebuild
cmake --build build -- -j8          # parallel build with 8 jobs
ccmake ..                           # interactive curses UI to set variables
cmake-gui                           # graphical UI
```

### 5. Install (optional)

```cmake
install(TARGETS cache_benchmark DESTINATION bin)
```

Then:

```bash
cmake --install build --prefix /usr/local
```

Copies the binary to `/usr/local/bin/cache_benchmark`.

---

## Key CMake vocabulary

| Term | Meaning |
|------|---------|
| **Target** | Something to build: an executable or a library |
| **Property** | Metadata on a target (compile flags, include dirs) |
| **Generator** | The build system CMake writes (Makefile, Ninja, VS) |
| **Cache variable** | A variable stored between CMake runs (`-DFOO=bar`) |
| `add_executable` | Declare an executable target |
| `target_compile_options` | Add compiler flags to a specific target |
| `target_include_directories` | Tell the compiler where headers live |
| `target_link_libraries` | Link a library to a target |
| `find_package` | Search for and import an external library |
