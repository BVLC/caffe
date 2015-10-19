# CNMeM Library

Simple library to help the Deep Learning frameworks manage CUDA memory. 

CNMeM is not intended to be a general purpose memory management library. It was designed as a simple
tool for applications which work on a limited number of large memory buffers.

CNMeM is mostly developed on Ubuntu Linux. It should support other operating systems as well. If you
encounter an issue with the library on other operating systems, please submit a bug (or a fix).

# Prerequisites

CNMeM relies on the CUDA toolkit. It uses C++ STL and the Pthread library on Linux. On Windows, it uses 
the native Win32 threading library. The build system uses CMake. The unit tests are written using
Google tests (but are not mandatory).

## CUDA

The CUDA toolkit is required. We recommend using CUDA >= 7.0 even if earlier versions will work. 
* Download from the [CUDA website](https://developer.nvidia.com/cuda-downloads)
* Follow the installation instructions
* Don't forget to set your path. For example:
  * `CUDA_HOME=/usr/local/cuda`
  * `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`

# Build CNMeM

## Grab the source

    % cd $HOME
    % git clone https://github.com/NVIDIA/cnmem.git cnmem

## Build CNMeM without the unit tests

    % cd cnmem
    % mkdir build
    % cd build
    % cmake ..
    % make

## Build CNMeM with the unit tests

To build the tests, you need to add an extra option to the cmake command.

    % cd cnmem
    % mkdir build
    % cd build
    % cmake -DWITH_TESTS=True ..
    % make

## Link with CNMeM

The source folder contains a header file 'include/cnmem.h' and the build directory contains the
library 'libcnmem.so', 'cnmem.lib/cnmem.dll' or 'libcnmem.dylib', depending on your operating 
system.

