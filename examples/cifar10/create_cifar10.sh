#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar10
DBTYPE=lmdb

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

if [ -x bin/convert_cifar_data ]
then
	bin/convert_cifar_data $DATA $EXAMPLE $DBTYPE
else
	bin/convert_cifar_data-d $DATA $EXAMPLE $DBTYPE
fi

echo "Computing image mean..."

if [ -x bin/compute_image_mean ]
then
	bin/compute_image_mean -backend=$DBTYPE \
	$EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto
else
	bin/compute_image_mean-d -backend=$DBTYPE \
	$EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto
fi	

echo "Done."
