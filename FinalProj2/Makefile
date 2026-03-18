# create exe project
project: main.cu kernel.cu cpumergesort.cpp
	nvcc -arch=sm_75 -std=c++17 -O2 main.cu kernel.cu cpumergesort.cpp -o project

# clean up the executable and text files
clean:
	rm -f project
	rm -f sort_comparison.txt
	rm -f gpu_sorted.txt
	rm -f cpu_sorted.txt
	rm -f unsorted.txt