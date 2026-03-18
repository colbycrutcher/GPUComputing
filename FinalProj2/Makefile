# create exe mergesort
mergesort: main.cu kernel.cu cpumergesort.cpp
	nvcc -arch=sm_75 -std=c++17 -O2 main.cu kernel.cu cpumergesort.cpp -o project

# clean up the executable and text files
clean:
	rm -f project
	rm -r sort_comparison.txt
	rm -r gpu_sorted.txt
	rm -r cpu_sorted.txt
	rm -r unsorted.txt