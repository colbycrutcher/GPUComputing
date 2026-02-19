## Lab6 Readme

## 1. In this program, how is the global synchronization ( or communication ) achieved ?

After finding the regional maxes uusing atomicMax, __syncthreads() is called to ensure there aren't any race conditions, ensuring all maxes are found before proceeding. Then we look for the max in the regional maxes.

## 2. Can we combine kernel global_max and kernel normalize into one kernel? What are the difficulties if we try to do that?



## 3. How needed data is passed from first kernel into second kernel? Did we reallocate memory for these data or did we make any copy of these data?

## 4. What conclusion(s) you can draw with regards to passing data around during multiple kernel launches? After first kernel is done, we launch second kernel, is the data in the global memory automatically wiped out or presistent before second kernel launch?
