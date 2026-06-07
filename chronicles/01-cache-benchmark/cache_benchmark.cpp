#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <chrono>

static constexpr size_t CACHE_LINE  = 64;
static constexpr size_t MAX_BYTES   = 256ULL * 1024 * 1024;
static constexpr size_t ACCESSES    = 1 << 26;

static double measure(char* buf, size_t buf_size, size_t stride) {
    size_t n = buf_size / stride;

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    // Fisher-Yates shuffle — breaks prefetcher
    uint64_t rng = 0xdeadbeefcafe1234ULL;
    for (size_t i = n - 1; i > 0; --i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        std::swap(idx[i], idx[rng % (i + 1)]);
    }

    // Embed pointer-chase into buffer
    for (size_t i = 0; i < n; ++i)
        *reinterpret_cast<size_t*>(buf + idx[i] * stride) = idx[(i + 1) % n] * stride;

    // Warm-up passes
    size_t pos = 0;
    for (int w = 0; w < 3; ++w)
        for (size_t i = 0; i < n; ++i)
            pos = *reinterpret_cast<size_t*>(buf + pos);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ACCESSES; ++i)
        pos = *reinterpret_cast<size_t*>(buf + pos);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (pos == static_cast<size_t>(-1)) std::cout << "";  // defeat DCE

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / static_cast<double>(ACCESSES);
}

static std::string fmt_size(size_t bytes) {
    if (bytes >= 1024 * 1024)
        return std::to_string(bytes >> 20) + " MB";
    return std::to_string(bytes >> 10) + " KB";
}

int main() {
    auto buf = std::make_unique<char[]>(MAX_BYTES);
    std::memset(buf.get(), 1, MAX_BYTES);

    static const size_t sizes[] = {
          8 * 1024,
         16 * 1024,
         32 * 1024,
         64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
          1 * 1024 * 1024,
          2 * 1024 * 1024,
          4 * 1024 * 1024,
          8 * 1024 * 1024,
         16 * 1024 * 1024,
         32 * 1024 * 1024,
         64 * 1024 * 1024,
        128 * 1024 * 1024,
    };

    std::cout << std::left
              << std::setw(12) << "Array size"
              << std::setw(12) << "Stride"
              << "Latency (ns/access)\n"
              << std::string(44, '-') << '\n';

    for (size_t arr : sizes) {
        size_t stride = CACHE_LINE;
        if (arr / stride < 64) stride = arr / 64;

        double lat = measure(buf.get(), arr, stride);

        std::cout << std::left
                  << std::setw(12) << fmt_size(arr)
                  << std::setw(12) << (std::to_string(stride) + " B")
                  << std::fixed << std::setprecision(2) << lat << '\n';
        std::cout.flush();
    }
}
