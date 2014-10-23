#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.

EXAMPLE=examples/siamese
DATA=data/mnist
BUILD=build/examples/mnist

BACKEND="lmdb"
BACKEND="files"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/mnist_train_${BACKEND}
rm -rf $EXAMPLE/mnist_test_${BACKEND}

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $EXAMPLE/mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $EXAMPLE/mnist_test_${BACKEND} --backend=${BACKEND}

for t in train test
do
    rm -f $EXAMPLE/mnist_${t}.txt

    for d in 0 1 2 3 4 5 6 7 8 9
    do
        for f in $EXAMPLE/mnist_${t}_${BACKEND}/$d/*.png
        do
            echo "$f $d" >>$EXAMPLE/mnist_${t}.txt
        done
    done

    shuf $EXAMPLE/mnist_${t}.txt >$EXAMPLE/mnist_${t}_p.txt
done

echo "Done."
