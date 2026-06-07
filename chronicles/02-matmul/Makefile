CXX      = clang++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra

matmul: matmul.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: matmul
	./matmul

clean:
	rm -f matmul

.PHONY: run clean
