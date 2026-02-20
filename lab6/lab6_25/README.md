# Lab6 Readme
## Riley Rudolfo, Colby Cruther, Ben


### 1. In this program, how is the global synchronization ( or communication ) achieved ?

In this program, gloal synchronization is achieved after launching the global_max kernel. The call to cudaDeviceSynchronize() forces the CPU to wait for all threads to finish executing and all the global memory has been updated. The cudaMemcpy also is a synchronizing operation because it has to wait for all prior GPU work to complete before moving data.

### 2. Can we combine kernel global_max and kernel normalize into one kernel? What are the difficulties if we try to do that?



### 3. How needed data is passed from first kernel into second kernel? Did we reallocate memory for these data or did we make any copy of these data?

### 4. What conclusion(s) you can draw with regards to passing data around during multiple kernel launches? After first kernel is done, we launch second kernel, is the data in the global memory automatically wiped out or presistent before second kernel launch?
