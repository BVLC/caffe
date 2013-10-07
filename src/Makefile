#
# The following defines a variable named "NAME" with a value of "myprogram". By convention,
# a lowercase prefix (in this case "program") and an uppercased suffix (in this case "NAME"), separated
# by an underscore is used to name attributes for a common element. Think of this like
# using program.NAME, program.C_SRCS, etc. There are no structs in Make, so we use this convention
# to keep track of attributes that all belong to the same target or program.  
#
PROJECT := caffe
NAME := lib$(PROJECT).so
TEST_NAME := test_$(PROJECT)
CXX_SRCS := $(shell find caffe ! -name "test_*.cpp" -name "*.cpp")
CU_SRCS := $(shell find caffe -name "*.cu")
TEST_SRCS := $(shell find caffe -name "test_*.cpp")
GTEST_SRC := gtest/gtest-all.cpp
PROGRAM_SRCS := $(shell find programs -name "*.cpp")
PROTO_SRCS := $(wildcard caffe/proto/*.proto)
PROTO_GEN_HEADER := ${PROTO_SRCS:.proto=.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}
PROTO_GEN_PY := ${PROTO_SRCS:.proto=_pb2.py}
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
CU_OBJS := ${CU_SRCS:.cu=.cuo}
PROGRAM_OBJS := ${PROGRAM_SRCS:.cpp=.o}
PROTO_OBJS := ${PROTO_SRCS:.proto=.pb.o}
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)
TEST_OBJS := ${TEST_SRCS:.cpp=.o}
GTEST_OBJ := ${GTEST_SRC:.cpp=.o}
TEST_BINS := ${TEST_OBJS:.o=.testbin}
PROGRAM_BINS :=${PROGRAM_OBJS:.o=.bin}

CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -arch=sm_20
MKL_DIR := /opt/intel/mkl

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

INCLUDE_DIRS := . /usr/local/include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS := . /usr/lib /usr/local/lib $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cuda cudart cublas protobuf glog mkl_rt mkl_intel_thread curand \
		leveldb snappy opencv_core opencv_highgui pthread tcmalloc
WARNINGS := -Wall

CXXFLAGS += -fPIC $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(WARNINGS)
NVCC = nvcc ${CXXFLAGS:-fPIC=-Xcompiler -fPIC} $(CPPFLAGS) $(CUDA_ARCH)

.PHONY: all test clean distclean linecount program

all: $(NAME)

linecount: clean
	cloc --read-lang-def=caffe.cloc caffe/

test: $(OBJS) $(GTEST_OBJ) $(TEST_BINS)

program: $(OBJS) $(PROGRAM_BINS)

runtest: test
	for testbin in $(TEST_BINS); do $$testbin; done

$(TEST_BINS): %.testbin : %.o
	$(CXX) -pthread $< $(OBJS) $(GTEST_OBJ) -o $@ $(LDFLAGS) $(WARNINGS)

$(PROGRAM_BINS): %.bin : %.o
	$(CXX) -pthread $< $(OBJS) -o $@ $(LDFLAGS) $(WARNINGS)

$(NAME): $(PROTO_GEN_CC) $(OBJS)
	$(LINK) -shared $(OBJS) -o $(NAME)

$(CU_OBJS): %.cuo: %.cu
	$(NVCC) -c $< -o $@

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=. --python_out=.

clean:
	@- $(RM) $(NAME) $(TEST_BINS) $(PROGRAM_BINS)
	@- $(RM) $(OBJS) $(TEST_OBJS) $(PROGRAM_OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)

distclean: clean
