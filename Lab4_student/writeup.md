Colby Crutcher, Riley Rudolfo

GPU Computing

Lab 4 Matrix Multiply


1. Read the provided code in src folder, specifically read the main function in the source file 
matrix_multiplication.cu. Answer the questions below, 

a. What tasks the program performs? 


 
b. What are the dependency C files and header files? How to call functions that is defined in another 
source file? 
 
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
 