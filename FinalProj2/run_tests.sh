#!/bin/bash

# Compile the code first to ensure we are testing the latest version
echo "Compiling the project..."
make clean
make

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Aborting tests."
    exit 1
fi

echo "Compilation successful. Starting benchmarks..."
echo "Results will be saved to sort_comparison.txt"
echo "--------------------------------------------------"

# Define the block size
BLOCK_SIZE=256

# Define an array of N values to test (powers of 4 to scale up quickly)
# Note: These must be multiples of the block size!
SIZES=(1024 4096 16384 65536 262144 1048576 4194304 16777216)

# Clear the output file so we start fresh
> sort_comparison.txt

# Loop through each size and run the executable
for N in "${SIZES[@]}"
do
    echo "Running test for N = $N..."
    ./mergesort $N $BLOCK_SIZE
done

echo "--------------------------------------------------"
echo "All tests complete! Run 'cat sort_comparison.txt' to view your benchmark data."