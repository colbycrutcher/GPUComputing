# Colby Crutcher, Ben Foster, Riley Rudolfo

Demo Video:

https://youtu.be/3UuTGXjHtrQ

* In the video the executable is called mergesort, but I changed it to 'project' per the rubric.
* The bash file is also different, as we now test different block sizes


# CUDA Merge Sort

This project implements a parallel merge sort on the GPU using a binary search-based approach[cite: 141, 253]. It compares the execution time of the GPU implementation against a standard sequential CPU merge sort to demonstrate parallel speedup.


# Results

|----------------|------------|----------------|----------------|-----------|
| Array Size (N) | Block Size | CPU Time (ms)  | GPU Time (ms)  |  Speedup  |
|----------------|------------|----------------|----------------|-----------|
| 1,024          |    128     |    0.0415 ms   |    0.0317 ms   |   1.31x   |
| 65,536         |    128     |    3.3914 ms   |    0.1132 ms   |  29.96x   |
| 1,048,576      |    128     |   54.4669 ms   |    1.6507 ms   |  33.00x   |
|----------------|------------|----------------|----------------|-----------|
| 1,024          |    256     |    0.0432 ms   |    0.0463 ms   |   0.93x   |
| 65,536         |    256     |    3.3841 ms   |    0.1708 ms   |  19.81x   |
| 1,048,576      |    256     |   55.0269 ms   |    2.4901 ms   |  22.10x   |
|----------------|------------|----------------|----------------|-----------|
| 1,024          |    512     |    0.0473 ms   |    0.1075 ms   |   0.44x   |
| 65,536         |    512     |    3.3668 ms   |    0.2898 ms   |  11.62x   |
| 1,048,576      |    512     |   54.9337 ms   |    4.2245 ms   |  13.00x   |
|----------------|------------|----------------|----------------|-----------|

* The GPU is slower for small arrays because the overhead of launching parallel kernels and managing device memory outweighs the actual computation time, whereas the CPU can process the data instantly within its high-speed local cache. Other than that we are getting massive speedups for larger data sets.



## How to Build

To compile the project, simply run the following command in your terminal:

make

Once compiled, you can run the program with its default settings (Array Size: 1024, Block Size: 256) by typing:

./project


Testing Custom Sizes:
You can also pass custom array sizes and block sizes directly in the terminal to see how the GPU handles larger datasets:

./project [ArraySize] [BlockSize]


Example (testing ~1 million elements):

./project 1048576 256


* Note - the total array size (N) must be evenly divisible by the number of blocks (numBlocks).
 
 I did this so that we don't have padding, and to avoid out of bounds queries.

 arg2 (Block Size) is The number of threads per CUDA block. This defines the initial "tile size" for the shared memory odd-even sort and the subsequent merge steps.


After that, you will see the results printed to sort_comparison.txt

ex: 

    Array Size (N): 1024 | Block Size: 256
    CPU Sequential Merge Sort Time: 0.062373 ms
    GPU Parallel Merge Sort Time: 0.016384 ms
    Speedup: 3.80695x


I also made a bash file that tests different array lengths. To use this, you may have to run:

chmod +x run_tests.sh

then,

./run_tests.sh

The results are also printed to sort_comparisons.txt


# Cleanup

To clean the file, run 'make clean'
 - This deletes the executable project, and the sort_comparison.txt