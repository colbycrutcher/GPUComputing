# Colby Crutcher, Ben Foster, Riley Rudolfo


# CUDA Merge Sort

This project implements a parallel merge sort on the GPU using a binary search-based approach[cite: 141, 253]. It compares the execution time of the GPU implementation against a standard sequential CPU merge sort to demonstrate parallel speedup.

## How to Build

To compile the project, simply run the following command in your terminal:

make

Once compiled, you can run the program with its default settings (Array Size: 1024, Block Size: 256) by typing:

./mergesort


Testing Custom Sizes:
You can also pass custom array sizes and block sizes directly in the terminal to see how the GPU handles larger datasets:

./mergesort [ArraySize] [BlockSize]


Example (testing ~1 million elements):

./mergesort 1048576 256


* Note - the total array size (N) must be evenly divisible by the number of blocks (numBlocks).


After that, you will see the results printed to sort_comparison.txt

ex: 

    Array Size (N): 1024 | Block Size: 256
    CPU Sequential Merge Sort Time: 0.062373 ms
    GPU Parallel Merge Sort Time: 0.016384 ms
    Speedup: 3.80695x


I also made a bash file that tests different array lengths. To use this, run

chmod +x run_tests.sh

then,

./run_tests.sh

The results are also printed to sort_comparisons.txt


# Cleanup

To clean the file, run 'make clean'
 - This deletes the executable mergesort, and the sort_comparison.txt