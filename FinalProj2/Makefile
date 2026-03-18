# The main target executable
mergesort: main.cu kernel.cu cpumergesort.cpp
	nvcc -arch=sm_75 -std=c++17 -O2 main.cu kernel.cu cpumergesort.cpp -o mergesort

# Clean up the executable
clean:
	rm -f mergesort
	rm -r sort_comparison.txt