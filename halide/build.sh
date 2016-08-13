#!/bin/bash

HALIDE_DIR=../../halide
CAFFE_DIR=../build/install

NAME=plip

#g++ ./generator/make_layers.cpp ./generator/GenGen.cpp -g -std=c++11 -fno-rtti -I${HALIDE_DIR}/include -L ${HALIDE_DIR}/bin -lHalide -lpthread -ldl  -lpthread -lz -fno-rtti -o 

#DYLD_LIBRARY_PATH=${HALIDE_DIR}/bin LD_LIBRARY_PATH=${HALIDE_DIR}/bin ./make_layers -g plip -o . target=cuda
#gcc -shared -o plip.so plip.o

# This refs caffe source
g++ ./wrappers/plip_wrapper.cpp ../build/halide/plip.o -I../build/halide/ -shared -g -std=c++11 -fPIC -fno-rtti -I${HALIDE_DIR}/include -I${CAFFE_DIR}/include -I/opt/cuda/include -L ${HALIDE_DIR}/bin -lHalide -L${CAFFE_DIR}/lib  -lpthread -ldl  -lpthread -lz -lglog -lgflags -lboost_system-mt -lcaffe -o plip_wrapper.so


