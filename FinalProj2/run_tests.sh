#!/bin/bash

echo "Compiling the project."
make clean
make

if [ $? -ne 0 ]; then
    echo "Compilation failed. Aborting tests."
    exit 1
fi

echo "Compilation successful. Starting benchmarks..."
echo "Results will be saved to sort_comparison.txt"
echo "--------------------------------------------------"


BLOCK_SIZE=256

SIZES=(1024 4096 16384 65536)


> sort_comparison.txt

for N in "${SIZES[@]}"
do
    echo "Running test for N = $N..."
    ./mergesort $N $BLOCK_SIZE
done

echo "--------------------------------------------------"
echo "All tests complete! Run 'cat sort_comparison.txt' to view your benchmark data."