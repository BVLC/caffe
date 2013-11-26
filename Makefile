# The makefile for caffe. Extremely hacky.
PROJECT := caffe
TEST_GPUID := 0

# define third-party library paths
# CHANGE YOUR CUDA PATH IF IT IS NOT THIS
CUDA_DIR := /usr/local/cuda
# CUDA_DIR := /Developer/NVIDIA/CUDA-5.5
# CHANGE YOUR CUDA ARCH IF IT IS NOT THIS
CUDA_ARCH := -arch=sm_30
# CHANGE YOUR MKL PATH IF IT IS NOT THIS
MKL_DIR := /opt/intel/mkl
# CHANGE YOUR MATLAB PATH IF IT IS NOT THIS
# your mex binary should be located at $(MATLAB_DIR)/bin/mex
MATLAB_DIR := /usr/local
# PUT ALL OTHER INCLUDE AND LIB DIRECTORIES HERE
INCLUDE_DIRS := /usr/local/include /usr/include/python2.7 \
    /usr/local/lib/python2.7/dist-packages/numpy/core/include
LIBRARY_DIRS := /usr/lib /usr/local/lib


##############################################################################
# After this line, things should happen automatically.
##############################################################################

# The target static library and shared library name
NAME := lib$(PROJECT).so
STATIC_NAME := lib$(PROJECT).a

##############################
# Get all source files
##############################
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src/caffe ! -name "test_*.cpp" -name "*.cpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/caffe -name "*.cu")
# TEST_SRCS are the test source files
TEST_SRCS := $(shell find src/caffe -name "test_*.cpp")
GTEST_SRC := src/gtest/gtest-all.cpp
# EXSAMPLE_SRCS are the source files for the example binaries
EXAMPLE_SRCS := $(shell find examples -name "*.cpp")
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRCS := $(wildcard src/caffe/proto/*.proto)
# PYCAFFE_SRC is the python wrapper for caffe
PYCAFFE_SRC := python/caffe/pycaffe.cpp
PYCAFFE_SO := python/caffe/pycaffe.so
# MATCAFFE_SRC is the matlab wrapper for caffe
MATCAFFE_SRC := matlab/caffe/matcaffe.cpp
MATCAFFE_SO := matlab/caffe/caffe

##############################
# Derive generated files
##############################
# The generated files for protocol buffers
PROTO_GEN_HEADER := ${PROTO_SRCS:.proto=.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}
PROTO_GEN_PY := ${PROTO_SRCS:.proto=_pb2.py}
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the test and example objects.
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
CU_OBJS := ${CU_SRCS:.cu=.cuo}
PROTO_OBJS := ${PROTO_GEN_CC:.cc=.o}
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)
# program and test objects
EXAMPLE_OBJS := ${EXAMPLE_SRCS:.cpp=.o}
TEST_OBJS := ${TEST_SRCS:.cpp=.o}
GTEST_OBJ := ${GTEST_SRC:.cpp=.o}
# program and test bins
EXAMPLE_BINS :=${EXAMPLE_OBJS:.o=.bin}
TEST_BINS := ${TEST_OBJS:.o=.testbin}

##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

INCLUDE_DIRS += ./src ./include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS += $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cudart cublas curand protobuf opencv_core opencv_highgui \
	glog mkl_rt mkl_intel_thread leveldb snappy pthread boost_system \
	opencv_imgproc
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall

# NOTE: on OS X, use clang++ to have NVCC play nice.
CXX := /usr/bin/c++
COMMON_FLAGS := -DNDEBUG $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC -O2 $(COMMON_FLAGS)
NVCCFLAGS := -ccbin=$(CXX) -Xcompiler -fPIC -O2 $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library)) \
		-Wl,-rpath,../lib/
PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))


##############################
# Define build targets
##############################
.PHONY: all test clean linecount examples pycaffe distribute

all: $(NAME) $(STATIC_NAME) examples

linecount: clean
	cloc --read-lang-def=caffe.cloc src/caffe/

test: $(TEST_BINS)

examples: $(EXAMPLE_BINS)

pycaffe: $(STATIC_NAME) $(PYCAFFE_SRC) $(PROTO_GEN_PY)
	$(CXX) -shared -o $(PYCAFFE_SO) $(PYCAFFE_SRC) \
		$(STATIC_NAME) $(CXXFLAGS) $(PYTHON_LDFLAGS)

matcaffe: $(STATIC_NAME) $(MATCAFFE_SRC)
	$(MATLAB_DIR)/bin/mex $(MATCAFFE_SRC) $(STATIC_NAME) \
		CXXFLAGS="\$$CXXFLAGS $(CXXFLAGS) $(WARNINGS)" \
		CXXLIBS="\$$CXXLIBS $(LDFLAGS)" \
		-o $(MATCAFFE_SO)

$(NAME): $(PROTO_OBJS) $(OBJS)
	$(CXX) -shared -o $(NAME) $(OBJS) $(LDFLAGS) $(WARNINGS)

$(STATIC_NAME): $(PROTO_OBJS) $(OBJS)
	ar rcs $(STATIC_NAME) $(PROTO_OBJS) $(OBJS)

runtest: test
	for testbin in $(TEST_BINS); do $$testbin $(TEST_GPUID); done

$(TEST_BINS): %.testbin : %.o $(GTEST_OBJ) $(STATIC_NAME)
	$(CXX) $< $(GTEST_OBJ) $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS)

$(EXAMPLE_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS)

$(OBJS): $(PROTO_GEN_CC)

$(EXAMPLE_OBJS): $(PROTO_GEN_CC)

$(CU_OBJS): %.cuo: %.cu
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

$(PROTO_GEN_PY): $(PROTO_SRCS)
	protoc --proto_path=src --python_out=python $(PROTO_SRCS)

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc --proto_path=src --cpp_out=src $(PROTO_SRCS)
	mkdir -p include/caffe/proto
	cp $(PROTO_GEN_HEADER) include/caffe/proto/

clean:
	@- $(RM) $(NAME) $(STATIC_NAME) $(TEST_BINS) $(EXAMPLE_BINS)
	@- $(RM) $(OBJS) $(TEST_OBJS) $(EXAMPLE_OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)
	@- $(RM) include/caffe/proto/caffe.pb.h
	@- $(RM) python/caffe/proto/caffe_pb2.py
	@- $(RM) -rf build

distribute: all
	mkdir build
	# add include
	cp -r include build/
	# add example binaries
	mkdir build/bin
	cp $(EXAMPLE_BINS) build/bin
	# add libraries
	mkdir build/lib
	cp $(NAME) build/lib
	cp $(STATIC_NAME) build/lib
	# add python - it's not the standard way, indeed...
	cp -r python build/python
