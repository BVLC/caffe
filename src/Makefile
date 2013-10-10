# The makefile for caffe. Extremely hack.
PROJECT := caffe
TEST_GPUID := 1

# The target static library and shared library name
NAME := lib$(PROJECT).so
STATIC_NAME := lib$(PROJECT).a
# All source files
CXX_SRCS := $(shell find caffe ! -name "test_*.cpp" -name "*.cpp")
CU_SRCS := $(shell find caffe -name "*.cu")
TEST_SRCS := $(shell find caffe -name "test_*.cpp")
GTEST_SRC := gtest/gtest-all.cpp
PROGRAM_SRCS := $(shell find programs -name "*.cpp")
PROTO_SRCS := $(wildcard caffe/proto/*.proto)
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
PROGRAM_OBJS := ${PROGRAM_SRCS:.cpp=.o}
TEST_OBJS := ${TEST_SRCS:.cpp=.o}
GTEST_OBJ := ${GTEST_SRC:.cpp=.o}
# program and test bins
PROGRAM_BINS :=${PROGRAM_OBJS:.o=.bin}
TEST_BINS := ${TEST_OBJS:.o=.testbin}

# define third-party library paths
CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -arch=sm_20
MKL_DIR := /opt/intel/mkl

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

# define inclue and libaries
INCLUDE_DIRS := . /usr/local/include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS := . /usr/lib /usr/local/lib $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cuda cudart cublas protobuf glog mkl_rt mkl_intel_thread curand \
		leveldb snappy opencv_core opencv_highgui pthread tcmalloc
WARNINGS := -Wall

COMMON_FLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC -O2 $(COMMON_FLAGS)
NVCCFLAGS := -Xcompiler -fPIC -O2 $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))

NVCC = nvcc $(NVCCFLAGS) $(CPPFLAGS) $(CUDA_ARCH)

.PHONY: all test clean distclean linecount program

all: $(NAME) $(STATIC_NAME) test program

linecount: clean
	cloc --read-lang-def=caffe.cloc caffe/

test: $(TEST_BINS)

program: $(PROGRAM_BINS)

$(NAME): $(PROTO_OBJS) $(OBJS)
	$(CXX) -shared $(OBJS) -o $(NAME) $(LDFLAGS) $(WARNINGS)

$(STATIC_NAME): $(PROTO_OBJS) $(OBJS)
	ar rcs $(STATIC_NAME) $(PROTO_OBJS) $(OBJS)

runtest: test
	for testbin in $(TEST_BINS); do $$testbin $(TEST_GPUID); done

$(TEST_BINS): %.testbin : %.o $(GTEST_OBJ) $(STATIC_NAME)
	$(CXX) $< $(GTEST_OBJ) $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS)

$(PROGRAM_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS)

$(CU_OBJS): %.cuo: %.cu
	$(NVCC) -c $< -o $@

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=. --python_out=.

clean:
	@- $(RM) $(NAME) $(STATIC_NAME) $(TEST_BINS) $(PROGRAM_BINS)
	@- $(RM) $(OBJS) $(TEST_OBJS) $(PROGRAM_OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)

distclean: clean
