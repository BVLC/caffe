# The makefile for caffe. Extremely hack.
PROJECT := caffe
TEST_GPUID := 1

# The target static library and shared library name
NAME := lib$(PROJECT).so
STATIC_NAME := lib$(PROJECT).a
# All source files
CXX_SRCS := $(shell find src/caffe ! -name "test_*.cpp" -name "*.cpp")
CU_SRCS := $(shell find src/caffe -name "*.cu")
TEST_SRCS := $(shell find src/caffe -name "test_*.cpp")
GTEST_SRC := src/gtest/gtest-all.cpp
EXAMPLE_SRCS := $(shell find examples -name "*.cpp")
PROTO_SRCS := $(wildcard src/caffe/proto/*.proto)
# The generated files for protocol buffers
PROTO_GEN_HEADER := ${PROTO_SRCS:.proto=.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}
PROTO_GEN_PY := ${PROTO_SRCS:.proto=_pb2.py}
# The objects that are needed to generate the library
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

# define third-party library paths
CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -arch=sm_30
MKL_DIR := /opt/intel/mkl

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

# define inclue and libaries
# We put src here just for gtest
INCLUDE_DIRS := ./src ./include /usr/local/include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS := /usr/lib /usr/local/lib $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cuda cudart cublas protobuf glog mkl_rt mkl_intel_thread curand \
		leveldb snappy pthread
WARNINGS := -Wall

COMMON_FLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC -O2 $(COMMON_FLAGS)
NVCCFLAGS := -Xcompiler -fPIC -O2 $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))

NVCC = nvcc $(NVCCFLAGS) $(CPPFLAGS) $(CUDA_ARCH)

.PHONY: all test clean distclean linecount examples distribute

all: $(NAME) $(STATIC_NAME) test examples

linecount: clean
	cloc --read-lang-def=caffe.cloc src/caffe/

test: $(TEST_BINS)

examples: $(EXAMPLE_BINS)

$(NAME): $(PROTO_OBJS) $(OBJS)
	$(CXX) -shared $(OBJS) -o $(NAME) $(LDFLAGS) $(WARNINGS)

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
	$(NVCC) -c $< -o $@

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc --proto_path=src --cpp_out=src --python_out=src $(PROTO_SRCS)
	mkdir -p include/caffe/proto
	cp $(PROTO_GEN_HEADER) include/caffe/proto/

clean:
	@- $(RM) $(NAME) $(STATIC_NAME) $(TEST_BINS) $(EXAMPLE_BINS)
	@- $(RM) $(OBJS) $(TEST_OBJS) $(EXAMPLE_OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)
	@- $(RM) -rf build

distclean: clean

distribute: all
	mkdir build
	cp -r include build/
	mkdir build/bin
	cp $(EXAMPLE_BINS) build/bin
	mkdir build/lib
	cp $(NAME) build/lib
	cp $(STATIC_NAME) build/lib
