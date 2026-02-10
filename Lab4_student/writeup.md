Colby Crutcher, Riley Rudolfo

GPU Computing

Lab 4 Matrix Multiply


1. Read the provided code in src folder, specifically read the main function in the source file 
matrix_multiplication.cu. Answer the questions below, 

a. What tasks the program performs? 

- Compares the speed of matrix multiplication on a CPU versus a GPU. It reads a square matrix from a file specified in the command line arguments and allocates memory on both the host (CPU) and the device (GPU).

 
b. What are the dependency C files and header files? How to call functions that is defined in another 
source file? 


- mul.h: Contains the prototype for the CPU-based multiplication function mul().

- arrayUtils.h: Contains prototypes for readNewArray, writeArray, and printArray.

- timing.h contains elapsedTime, timeCost, and currentTime

- matrix_multiplication.cu: The main entry point containing the CUDA kernel and the logic for the benchmarking loop.

- timing.c: Contains the implementation for measuring time using gettimeofday.

- mul.c: Implements the sequential triple-nested loop for matrix multiplication.

- arrayUtils.c: Implements the logic for reading matrices from files using fscanf and writing them using fprintf

C files:
 
2. Based upon the lecture notes, please write the simple kernel function to perform matrix multiplication 
on GPU, on top of the source file matrix_multiplication.cu. 


3.  Explore the Makefile, how shall we jointly compile .cu and .c files in a single project? 
 
4. Check the APIs Docs and find out what cudaEventRecord() and cudaEventSynchronize() do? 
 
5. In the main function, what is the equation that the program uses to compute the throughput (in unit of 
GLOPS)? Please interpret the equation. 
 

6. Run your program after you finish your kernel on the dataset 1024.mat and 2048.mat, how much 
speedups do you obtain compared with CPU time cost? What are the GPU throughput and CPU 
throughput you observed in these cases? 
 
7. Can you find some specification data about the peak performance (GFLOPS) for the GPU that we are 
using in the Lab (NVIDIA RTX 3070) ? Is the GPU device throughput that you observed in step 6 close 
to their Peak performance? Guess the reason why they are close or why they are far away? 
 