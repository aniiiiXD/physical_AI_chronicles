CXX     = clang++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

cache_benchmark: cache_benchmark.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: cache_benchmark
	./cache_benchmark

clean:
	rm -f cache_benchmark

.PHONY: run clean
