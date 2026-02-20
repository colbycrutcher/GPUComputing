# Lab6 Readme
## Riley Rudolfo, Colby Cruther, Ben Foster


### 1. In this program, how is the global synchronization ( or communication ) achieved ?

In this program, gloal synchronization is achieved after launching the global_max kernel. The call to cudaDeviceSynchronize() forces the CPU to wait for all threads to finish executing and all the global memory has been updated. The cudaMemcpy also is a synchronizing operation because it has to wait for all prior GPU work to complete before moving data.

### 2. Can we combine kernel global_max and kernel normalize into one kernel? What are the difficulties if we try to do that?

We can't combine global_max and normalize into a single kernel because the normalized kernel needs the final global max, which is only known after a reduction across all blocks. This is difficult because of enforcing correct global synchronization and visibility of the max before all the threads perform the division.

### 3. How needed data is passed from first kernel into second kernel? Did we reallocate memory for these data or did we make any copy of these data?

The needed data is passed from global_max to the normalize kernel through the device global memory. The global_max kernel writes its result into the d_gl_max, and normalize reads from the pointer that's pointing at d_gl_max. We don't reallocate between the kernels because it's only allocated once with cudaMalloc and then reused. The host only uses cudaDeviceSynchonize() to enforce correct order by having all the data get to the second kernel before looking at the completed results. 

### 4. What conclusion(s) you can draw with regards to passing data around during multiple kernel launches? After first kernel is done, we launch second kernel, is the data in the global memory automatically wiped out or presistent before second kernel launch?
