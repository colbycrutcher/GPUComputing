When you’re sure you have correct code, prepare a simple chart like the one below, using the tool of your choice. First, do a “make clean”, and then rebuild your code with the following options:

make dbg="" cuda_dbg="" opt="-O3"

Test your performance with the following command lines, replacing threadsPerBlock and n with the values given in the chart:
./jacobi threadsPerBlock 1000 n n


Sample run to check the output of different runs:
The diff command below will compare the cuda output with the output of the sequential code.


./jacobi 512 100 1600 1600 p > pout2
./jacobi 0 100 1600 1600 p > sout1
diff pout2 sout1  

