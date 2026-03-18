# block sizes
BLOCK_SIZES=(128 256 512)
# array sizes
SIZES=(1024 65536 1048576)

for B in "${BLOCK_SIZES[@]}"
do
    for N in "${SIZES[@]}"
    do
        echo "Testing N=$N with BlockSize=$B"
        ./project $N $B
    done
done