# The makefile for caffe. Extremely hacky.
PROJECT := caffe
TEST_GPUID := 0

include Makefile.config

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
# HXX_SRCS are the header files
HXX_SRCS := $(shell find include/caffe ! -name "*.hpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/caffe -name "*.cu")
# TEST_SRCS are the test source files
TEST_SRCS := $(shell find src/caffe -name "test_*.cpp")
GTEST_SRC := src/gtest/gtest-all.cpp
# TEST_HDRS are the test header files
TEST_HDRS := $(shell find src/caffe -name "test_*.hpp")
# EXAMPLE_SRCS are the source files for the example binaries
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
CUDA_LIB_DIR := $(CUDA_DIR)/lib64 $(CUDA_DIR)/lib
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

INCLUDE_DIRS += ./src ./include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS += $(CUDA_LIB_DIR) $(MKL_LIB_DIR) /usr/lib/atlas-base
LIBRARIES := cudart cublas curand protobuf \
        opencv_core opencv_highgui opencv_imgproc \
	glog \
	atlas cblas \
	leveldb snappy pthread boost_system 
	# mkl_rt mkl_intel_thread 
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall

COMMON_FLAGS := -DNDEBUG -O2 $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
NVCCFLAGS := -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library)) -Wl,-rpath=/usr/lib/atlas-base
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

$(TEST_BINS): %.testbin : %.o $(GTEST_OBJ) $(STATIC_NAME) $(TEST_HDRS)
	$(CXX) $< $(GTEST_OBJ) $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS)

$(EXAMPLE_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS)

$(OBJS): $(PROTO_GEN_CC) $(HXX_SRCS)

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
