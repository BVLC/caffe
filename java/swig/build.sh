#!/bin/bash

PROJECT=caffe
BASE_DIR=../../
CC=gcc
CXX=g++
CUDA_INCLUDE_DIR=/usr/local/cuda/include
JDK_INCLUDE_DIR=/usr/lib/jvm/java-7-openjdk-amd64/include
JDK_OS_INCLUDE_DIR=$JDK_INCLUDE_DIR/linux
PACKAGE=org.berkeleyvision.caffe
OUT_DIR=gen/org/berkeleyvision/caffe

cp -R $BASE_DIR/include/caffe .
cp -R $BASE_DIR/build/src/caffe/proto caffe
mkdir -p $OUT_DIR

swig -c++ -java -cpperraswarn -outdir $OUT_DIR -package $PACKAGE $PROJECT.i 
$CXX -fPIC -c ${PROJECT}_wrap.cxx -I. -I$BASE_DIR/build/src -I$BASE_DIR/build/src/$PROJECT -I$CUDA_INCLUDE_DIR -I$JDK_INCLUDE_DIR -I$JDK_OS_INCLUDE_DIR

OBJS=`find $BASE_DIR/build/src/caffe -name "*.o"`
CU_OBJS=`find $BASE_DIR/build/src/caffe -name "*.cuo"`
#echo $OBJS
#echo $CU_OBJS
$CXX -shared $OBJS $CU_OBJS ${PROJECT}_wrap.o -o lib${PROJECT}.so
