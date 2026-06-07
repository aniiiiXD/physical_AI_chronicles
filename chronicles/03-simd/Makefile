CXX      = clang++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

simd_bench: simd_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: simd_bench
	./simd_bench

clean:
	rm -f simd_bench

.PHONY: run clean
