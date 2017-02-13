cmake_minimum_required(VERSION 3.1)
project(CaffeModularLayer)

find_package(Caffe REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${Caffe_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# This example acts as a standalone example, but Caffe insists on building
# the cpp file. We define COMPILE_MODULAR_LAYER to let the cpp file know
# it is being build from this CMakeLists.txt.
add_definitions(-DCOMPILE_MODULAR_LAYER)

# These variables are used as prefix and suffix for the generated module.
# They can be overwritten as below, but make sure they match with what
# is expected on the Caffe side (either through cmake or Makefile.config).
#set(CMAKE_SHARED_MODULE_PREFIX "lib")
#set(CMAKE_SHARED_MODULE_SUFFIX ".so")

# Use MODULE here to generate a module which we can load during runtime.
add_library(modular_layer MODULE src/modular_layer.cpp)
target_link_libraries(modular_layer ${Caffe_LIBRARIES} ${CUDA_LIBRARIES})
