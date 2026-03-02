Colby Crutcher, Riley Rudolfo

Running this program:

1. Run make to build executable 'reduce'
2. Run './reduce blockWidth numElementsInput p' 
3. Run 'make clean' to clear binary and object file and prepare for recompilation


______________________________________________________________________________
|Input Size                     | 1048576 | 16777216 |  67108864 | 134217728 |
|-------------------------------|---------|----------|-----------|-----------|
|Block Dimensions               | 1 x 1024| 1 x 1024 |  1 x 1024 |  1 x 1024 |
|-------------------------------|---------|----------|-----------|-----------|
|T1:time cost for reduce2 (s)   | 0.000260| 0.001512 | 0.005485  | 0.010775  |
|-------------------------------|---------|----------|-----------|-----------|
|T2:time cost for reduce3 (s)   |0.000107 | 0.000937 | 0.003355  | 0.006531  |
|-------------------------------|---------|----------|-----------|-----------|
|Speedup = T1/T2                | 2.4299  | 1.61366  | 1.634873  | 1.6498239 |
|_______________________________|_________|__________|___________|___________|
* Output prints in seconds, not milliseconds. If you want milliseconds divide by 1000


reduce3 kernel, which utilizes sequential addressing, provides significant performance advantages over the interleaved addressing used in reduce2.  By keeping active threads contiguous (such as threads 0 through 7 working tightly together), reduce3 eliminates the condition divergence and shared memory bank conflicts that slow down reduce2. Also, reduce reduces the number of idle threads by ensuring all launched threads are active during the first step, whereas reduce2 leaves half of its threads idle immediately due to its execution pattern
+3