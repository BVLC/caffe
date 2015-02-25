# The makefile for caffe. Pretty hacky.
PROJECT := caffe

CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

BUILD_DIR_LINK := $(BUILD_DIR)
RELEASE_BUILD_DIR := .$(BUILD_DIR)_release
DEBUG_BUILD_DIR := .$(BUILD_DIR)_debug

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	BUILD_DIR := $(DEBUG_BUILD_DIR)
	OTHER_BUILD_DIR := $(RELEASE_BUILD_DIR)
else
	BUILD_DIR := $(RELEASE_BUILD_DIR)
	OTHER_BUILD_DIR := $(DEBUG_BUILD_DIR)
endif

# The target shared library and static library name
LIB_BUILD_DIR := $(BUILD_DIR)/lib
NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).a

##############################
# Get all source files
##############################
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cpp" -name "*.cpp")
# HXX_SRCS are the header files
HXX_SRCS := $(shell find include/$(PROJECT) ! -name "test_*.hpp" -name "*.hpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cu" -name "*.cu")
# TEST_SRCS are the test source files
TEST_MAIN_SRC := src/$(PROJECT)/test/test_caffe_main.cpp
TEST_SRCS := $(shell find src/$(PROJECT) -name "test_*.cpp")
TEST_SRCS := $(filter-out $(TEST_MAIN_SRC), $(TEST_SRCS))
TEST_CU_SRCS := $(shell find src/$(PROJECT) -name "test_*.cu")
GTEST_SRC := src/gtest/gtest-all.cpp
# TEST_HXX_SRCS are the test header files
TEST_HXX_SRCS := $(shell find include/$(PROJECT) -name "test_*.hpp")
# TOOL_SRCS are the source files for the tool binaries
TOOL_SRCS := $(shell find tools -name "*.cpp")
# EXAMPLE_SRCS are the source files for the example binaries
EXAMPLE_SRCS := $(shell find examples -name "*.cpp")
# BUILD_INCLUDE_DIR contains any generated header files we want to include.
BUILD_INCLUDE_DIR := $(BUILD_DIR)/src
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRC_DIR := src/$(PROJECT)/proto
PROTO_SRCS := $(wildcard $(PROTO_SRC_DIR)/*.proto)
# PROTO_BUILD_DIR will contain the .cc and obj files generated from
# PROTO_SRCS; PROTO_BUILD_INCLUDE_DIR will contain the .h header files
PROTO_BUILD_DIR := $(BUILD_DIR)/$(PROTO_SRC_DIR)
PROTO_BUILD_INCLUDE_DIR := $(BUILD_INCLUDE_DIR)/$(PROJECT)/proto
# NONGEN_CXX_SRCS includes all source/header files except those generated
# automatically (e.g., by proto).
NONGEN_CXX_SRCS := $(shell find \
	src/$(PROJECT) \
	include/$(PROJECT) \
	python/$(PROJECT) \
	matlab/$(PROJECT) \
	examples \
	tools \
	-name "*.cpp" -or -name "*.hpp" -or -name "*.cu" -or -name "*.cuh")
LINT_SCRIPT := scripts/cpp_lint.py
LINT_OUTPUT_DIR := $(BUILD_DIR)/.lint
LINT_EXT := lint.txt
LINT_OUTPUTS := $(addsuffix .$(LINT_EXT), $(addprefix $(LINT_OUTPUT_DIR)/, $(NONGEN_CXX_SRCS)))
EMPTY_LINT_REPORT := $(BUILD_DIR)/.$(LINT_EXT)
NONEMPTY_LINT_REPORT := $(BUILD_DIR)/$(LINT_EXT)
# PY$(PROJECT)_SRC is the python wrapper for $(PROJECT)
PY$(PROJECT)_SRC := python/$(PROJECT)/_$(PROJECT).cpp
PY$(PROJECT)_HXX_SRC := python/$(PROJECT)/_$(PROJECT).hpp
PY$(PROJECT)_SO := python/$(PROJECT)/_$(PROJECT).so
# MAT$(PROJECT)_SRC is the matlab wrapper for $(PROJECT)
MAT$(PROJECT)_SRC := matlab/$(PROJECT)/mat$(PROJECT).cpp
ifneq ($(MATLAB_DIR),)
	MAT_SO_EXT := $(shell $(MATLAB_DIR)/bin/mexext)
endif
MAT$(PROJECT)_SO := matlab/$(PROJECT)/$(PROJECT).$(MAT_SO_EXT)

##############################
# Derive generated files
##############################
# The generated files for protocol buffers
PROTO_GEN_HEADER_SRCS := $(addprefix $(PROTO_BUILD_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
PROTO_GEN_HEADER := $(addprefix $(PROTO_BUILD_INCLUDE_DIR)/, \
		$(notdir ${PROTO_SRCS:.proto=.pb.h}))
HXX_SRCS += $(PROTO_GEN_HEADER)
PROTO_GEN_CC := $(addprefix $(BUILD_DIR)/, ${PROTO_SRCS:.proto=.pb.cc})
PY_PROTO_BUILD_DIR := python/$(PROJECT)/proto
PY_PROTO_INIT := python/$(PROJECT)/proto/__init__.py
PROTO_GEN_PY := $(foreach file,${PROTO_SRCS:.proto=_pb2.py}, \
		$(PY_PROTO_BUILD_DIR)/$(notdir $(file)))
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the tool, example, and test objects.
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.cuo})
PROTO_OBJS := ${PROTO_GEN_CC:.cc=.o}
OBJ_BUILD_DIR := $(BUILD_DIR)/src/$(PROJECT)
LAYER_BUILD_DIR := $(OBJ_BUILD_DIR)/layers
UTIL_BUILD_DIR := $(OBJ_BUILD_DIR)/util
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)
# tool, example, and test objects
TOOL_OBJS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o})
TOOL_BUILD_DIR := $(BUILD_DIR)/tools
TEST_BUILD_DIR := $(BUILD_DIR)/src/$(PROJECT)/test
TEST_CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cpp=.o})
TEST_CU_OBJS := $(addprefix $(BUILD_DIR)/, ${TEST_CU_SRCS:.cu=.cuo})
TEST_OBJS := $(TEST_CXX_OBJS) $(TEST_CU_OBJS)
GTEST_OBJ := $(addprefix $(BUILD_DIR)/, ${GTEST_SRC:.cpp=.o})
GTEST_BUILD_DIR := $(dir $(GTEST_OBJ))
EXAMPLE_OBJS := $(addprefix $(BUILD_DIR)/, ${EXAMPLE_SRCS:.cpp=.o})
EXAMPLE_BUILD_DIR := $(BUILD_DIR)/examples
EXAMPLE_BUILD_DIRS := $(EXAMPLE_BUILD_DIR)
EXAMPLE_BUILD_DIRS += $(foreach obj,$(EXAMPLE_OBJS),$(dir $(obj)))
# tool, example, and test bins
TOOL_BINS := ${TOOL_OBJS:.o=.bin}
EXAMPLE_BINS := ${EXAMPLE_OBJS:.o=.bin}
# symlinks to tool bins without the ".bin" extension
TOOL_BIN_LINKS := ${TOOL_BINS:.bin=}
# Put the test binaries in build/test for convenience.
TEST_BIN_DIR := $(BUILD_DIR)/test
TEST_CU_BINS := $(addsuffix .testbin,$(addprefix $(TEST_BIN_DIR)/, \
		$(foreach obj,$(TEST_CU_OBJS),$(basename $(notdir $(obj))))))
TEST_CXX_BINS := $(addsuffix .testbin,$(addprefix $(TEST_BIN_DIR)/, \
		$(foreach obj,$(TEST_CXX_OBJS),$(basename $(notdir $(obj))))))
TEST_BINS := $(TEST_CXX_BINS) $(TEST_CU_BINS)
TEST_ALL_BIN := $(TEST_BIN_DIR)/test_all.testbin

##############################
# Derive compiler warning dump locations
##############################
WARNS_EXT := warnings.txt
CXX_WARNS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o.$(WARNS_EXT)})
CU_WARNS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.cuo.$(WARNS_EXT)})
TOOL_WARNS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o.$(WARNS_EXT)})
EXAMPLE_WARNS := $(addprefix $(BUILD_DIR)/, ${EXAMPLE_SRCS:.cpp=.o.$(WARNS_EXT)})
TEST_WARNS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cpp=.o.$(WARNS_EXT)})
TEST_CU_WARNS := $(addprefix $(BUILD_DIR)/, ${TEST_CU_SRCS:.cu=.cuo.$(WARNS_EXT)})
ALL_CXX_WARNS := $(CXX_WARNS) $(TOOL_WARNS) $(EXAMPLE_WARNS) $(TEST_WARNS)
ALL_CU_WARNS := $(CU_WARNS) $(TEST_CU_WARNS)
ALL_WARNS := $(ALL_CXX_WARNS) $(ALL_CU_WARNS)

EMPTY_WARN_REPORT := $(BUILD_DIR)/.$(WARNS_EXT)
NONEMPTY_WARN_REPORT := $(BUILD_DIR)/$(WARNS_EXT)

##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include

CUDA_LIB_DIR :=
# add <cuda>/lib64 only if it exists
ifneq ("$(wildcard $(CUDA_DIR)/lib64)","")
	CUDA_LIB_DIR += $(CUDA_DIR)/lib64
endif
CUDA_LIB_DIR += $(CUDA_DIR)/lib

INCLUDE_DIRS += $(BUILD_INCLUDE_DIR) ./src ./include
ifneq ($(CPU_ONLY), 1)
	INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
	LIBRARY_DIRS += $(CUDA_LIB_DIR)
	LIBRARIES := cudart cublas curand
endif
LIBRARIES += glog gflags protobuf leveldb snappy \
	lmdb \
	boost_system \
	hdf5_hl hdf5 \
	opencv_core opencv_highgui opencv_imgproc pthread
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall -Wno-sign-compare

##############################
# Set build directories
##############################

DISTRIBUTE_SUBDIRS := $(DISTRIBUTE_DIR)/bin $(DISTRIBUTE_DIR)/lib
DIST_ALIASES := dist
ifneq ($(strip $(DISTRIBUTE_DIR)),distribute)
		DIST_ALIASES += distribute
endif

ALL_BUILD_DIRS := $(sort \
		$(BUILD_DIR) $(LIB_BUILD_DIR) $(OBJ_BUILD_DIR) \
		$(LAYER_BUILD_DIR) $(UTIL_BUILD_DIR) $(TOOL_BUILD_DIR) \
		$(TEST_BUILD_DIR) $(TEST_BIN_DIR) $(GTEST_BUILD_DIR) \
		$(EXAMPLE_BUILD_DIRS) \
		$(LINT_OUTPUT_DIR) \
		$(PROTO_BUILD_DIR) $(PROTO_BUILD_INCLUDE_DIR) $(PY_PROTO_BUILD_DIR) \
		$(DISTRIBUTE_SUBDIRS))

##############################
# Set directory for Doxygen-generated documentation
##############################
DOXYGEN_CONFIG_FILE ?= ./.Doxyfile
# should be the same as OUTPUT_DIRECTORY in the .Doxyfile
DOXYGEN_OUTPUT_DIR ?= ./doxygen
DOXYGEN_COMMAND ?= doxygen
# All the files that might have Doxygen documentation.
DOXYGEN_SOURCES := $(shell find \
	src/$(PROJECT) \
	include/$(PROJECT) \
	python/ \
	matlab/ \
	examples \
	tools \
	-name "*.cpp" -or -name "*.hpp" -or -name "*.cu" -or -name "*.cuh" -or \
        -name "*.py" -or -name "*.m")
DOXYGEN_SOURCES += $(DOXYGEN_CONFIG_FILE)


##############################
# Configure build
##############################

# Determine platform
UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LINUX := 1
else ifeq ($(UNAME), Darwin)
	OSX := 1
endif

ifeq ($(LINUX), 1)
	CXX ?= /usr/bin/g++
	GCCVERSION := $(shell $(CXX) -dumpversion | cut -f1,2 -d.)
	# older versions of gcc are too dumb to build boost with -Wuninitalized
	ifeq ($(shell echo $(GCCVERSION) \< 4.6 | bc), 1)
		WARNINGS += -Wno-uninitialized
	endif
	# boost::thread is reasonably called boost_thread (compare OS X)
	LIBRARIES += boost_thread
endif

# OS X:
# clang++ instead of g++
# libstdc++ instead of libc++ for CUDA compatibility on 10.9
ifeq ($(OSX), 1)
	CXX := /usr/bin/clang++
	# clang throws this warning for cuda headers
	WARNINGS += -Wno-unneeded-internal-declaration
	ifneq ($(findstring 10.9, $(shell sw_vers -productVersion)),)
		CXXFLAGS += -stdlib=libstdc++
		LINKFLAGS += -stdlib=libstdc++
	endif
	# boost::thread is called boost_thread-mt to mark multithreading on OS X
	LIBRARIES += boost_thread-mt
endif

# Custom compiler
ifdef CUSTOM_CXX
	CXX := $(CUSTOM_CXX)
endif

# Debugging
ifeq ($(DEBUG), 1)
	COMMON_FLAGS += -DDEBUG -g -O0
	NVCCFLAGS += -G
else
	COMMON_FLAGS += -DNDEBUG -O2
endif

# cuDNN acceleration configuration.
ifeq ($(USE_CUDNN), 1)
	LIBRARIES += cudnn
	COMMON_FLAGS += -DUSE_CUDNN
endif

# CPU-only configuration
ifeq ($(CPU_ONLY), 1)
	OBJS := $(PROTO_OBJS) $(CXX_OBJS)
	TEST_OBJS := $(TEST_CXX_OBJS)
	TEST_BINS := $(TEST_CXX_BINS)
	ALL_WARNS := $(ALL_CXX_WARNS)
	TEST_FILTER := --gtest_filter="-*GPU*"
	COMMON_FLAGS += -DCPU_ONLY
endif

# BLAS configuration (default = ATLAS)
BLAS ?= atlas
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKL_DIR ?= /opt/intel/mkl
	BLAS_INCLUDE ?= $(MKL_DIR)/include
	BLAS_LIB ?= $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64
else ifeq ($(BLAS), open)
	# OpenBLAS
	LIBRARIES += openblas
else
	# ATLAS
	ifeq ($(LINUX), 1)
		ifeq ($(BLAS), atlas)
			# Linux simply has cblas and atlas
			LIBRARIES += cblas atlas
		endif
	else ifeq ($(OSX), 1)
		# OS X packages atlas as the vecLib framework
		BLAS_INCLUDE ?= /System/Library/Frameworks/vecLib.framework/Versions/Current/Headers/
		LIBRARIES += cblas
		LDFLAGS += -framework vecLib
	endif
endif
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
# mex may invoke an older gcc that is too liberal with -Wuninitalized
MATLAB_CXXFLAGS := $(CXXFLAGS) -Wno-uninitialized
LINKFLAGS += -fPIC $(COMMON_FLAGS) $(WARNINGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))
PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))

# 'superclean' target recursively* deletes all files ending with an extension
# in $(SUPERCLEAN_EXTS) below.  This may be useful if you've built older
# versions of Caffe that do not place all generated files in a location known
# to the 'clean' target.
#
# 'supercleanlist' will list the files to be deleted by make superclean.
#
# * Recursive with the exception that symbolic links are never followed, per the
# default behavior of 'find'.
SUPERCLEAN_EXTS := .so .a .o .bin .testbin .pb.cc .pb.h _pb2.py .cuo

##############################
# Define build targets
##############################
.PHONY: all test clean docs linecount lint lintclean tools examples $(DIST_ALIASES) \
	py mat py$(PROJECT) mat$(PROJECT) proto runtest \
	superclean supercleanlist supercleanfiles warn everything

all: $(NAME) $(STATIC_NAME) tools examples

everything: all py$(PROJECT) mat$(PROJECT) test warn lint runtest

linecount:
	cloc --read-lang-def=$(PROJECT).cloc \
		src/$(PROJECT) include/$(PROJECT) tools examples \
		python matlab

lint: $(EMPTY_LINT_REPORT)

lintclean:
	@ $(RM) -r $(LINT_OUTPUT_DIR) $(EMPTY_LINT_REPORT) $(NONEMPTY_LINT_REPORT)

docs: $(DOXYGEN_OUTPUT_DIR)
	@ cd ./docs ; ln -sfn ../$(DOXYGEN_OUTPUT_DIR)/html doxygen

$(DOXYGEN_OUTPUT_DIR): $(DOXYGEN_CONFIG_FILE) $(DOXYGEN_SOURCES)
	$(DOXYGEN_COMMAND) $(DOXYGEN_CONFIG_FILE)

$(EMPTY_LINT_REPORT): $(LINT_OUTPUTS) | $(BUILD_DIR)
	@ cat $(LINT_OUTPUTS) > $@
	@ if [ -s "$@" ]; then \
		cat $@; \
		mv $@ $(NONEMPTY_LINT_REPORT); \
		echo "Found one or more lint errors."; \
		exit 1; \
	  fi; \
	  $(RM) $(NONEMPTY_LINT_REPORT); \
	  echo "No lint errors!";

$(LINT_OUTPUTS): $(LINT_OUTPUT_DIR)/%.lint.txt : % $(LINT_SCRIPT) | $(LINT_OUTPUT_DIR)
	@ mkdir -p $(dir $@)
	@ python $(LINT_SCRIPT) $< 2>&1 \
		| grep -v "^Done processing " \
		| grep -v "^Total errors found: 0" \
		> $@ \
		|| true

test: $(TEST_ALL_BIN) $(TEST_BINS)

tools: $(TOOL_BINS) $(TOOL_BIN_LINKS)

examples: $(EXAMPLE_BINS)

py$(PROJECT): py

py: $(PY$(PROJECT)_SO) $(PROTO_GEN_PY)

$(PY$(PROJECT)_SO): $(STATIC_NAME) $(PY$(PROJECT)_SRC) $(PY$(PROJECT)_HXX_SRC)
	$(CXX) -shared -o $@ $(PY$(PROJECT)_SRC) \
		$(STATIC_NAME) $(LINKFLAGS) $(PYTHON_LDFLAGS)
	@ echo

mat$(PROJECT): mat

mat: $(MAT$(PROJECT)_SO)

$(MAT$(PROJECT)_SO): $(MAT$(PROJECT)_SRC) $(STATIC_NAME)
	@ if [ -z "$(MATLAB_DIR)" ]; then \
		echo "MATLAB_DIR must be specified in $(CONFIG_FILE)" \
			"to build mat$(PROJECT)."; \
		exit 1; \
	fi
	$(MATLAB_DIR)/bin/mex $(MAT$(PROJECT)_SRC) \
			CXX="$(CXX)" \
			CXXFLAGS="\$$CXXFLAGS $(MATLAB_CXXFLAGS)" \
			CXXLIBS="\$$CXXLIBS $(STATIC_NAME) $(LDFLAGS)" -output $@
	@ echo

runtest: $(TEST_ALL_BIN)
	$(TEST_ALL_BIN) $(TEST_GPUID) --gtest_shuffle $(TEST_FILTER)

warn: $(EMPTY_WARN_REPORT)

$(EMPTY_WARN_REPORT): $(ALL_WARNS) | $(BUILD_DIR)
	@ cat $(ALL_WARNS) > $@
	@ if [ -s "$@" ]; then \
		cat $@; \
		mv $@ $(NONEMPTY_WARN_REPORT); \
		echo "Compiler produced one or more warnings."; \
		exit 1; \
	  fi; \
	  $(RM) $(NONEMPTY_WARN_REPORT); \
	  echo "No compiler warnings!";

$(ALL_CXX_WARNS): %.o.$(WARNS_EXT) : %.o

$(ALL_CU_WARNS): %.cuo.$(WARNS_EXT) : %.cuo

$(BUILD_DIR_LINK): $(BUILD_DIR)/.linked

# Create a target ".linked" in this BUILD_DIR to tell Make that the "build" link
# is currently correct, then delete the one in the OTHER_BUILD_DIR in case it
# exists and $(DEBUG) is toggled later.
$(BUILD_DIR)/.linked:
	@ mkdir -p $(BUILD_DIR)
	@ $(RM) $(OTHER_BUILD_DIR)/.linked
	@ $(RM) -r $(BUILD_DIR_LINK)
	@ ln -s $(BUILD_DIR) $(BUILD_DIR_LINK)
	@ touch $@

$(ALL_BUILD_DIRS): | $(BUILD_DIR_LINK)
	@ mkdir -p $@

$(NAME): $(PROTO_OBJS) $(OBJS) | $(LIB_BUILD_DIR)
	$(CXX) -shared -o $@ $(OBJS) $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(STATIC_NAME): $(PROTO_OBJS) $(OBJS) | $(LIB_BUILD_DIR)
	ar rcs $@ $(PROTO_OBJS) $(OBJS)
	@ echo

$(TEST_BUILD_DIR)/%.o: src/$(PROJECT)/test/%.cpp $(HXX_SRCS) $(TEST_HXX_SRCS) \
		| $(TEST_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(TEST_BUILD_DIR)/%.cuo: src/$(PROJECT)/test/%.cu $(HXX_SRCS) $(TEST_HXX_SRCS) \
		| $(TEST_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(TEST_ALL_BIN): $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(STATIC_NAME) \
		-o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(TEST_CU_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_BUILD_DIR)/%.cuo $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) $(STATIC_NAME) \
		-o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(TEST_CXX_BINS): $(TEST_BIN_DIR)/%.testbin: $(TEST_BUILD_DIR)/%.o $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) $(STATIC_NAME) \
		-o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

# Target for extension-less symlinks to tool binaries with extension '*.bin'.
$(TOOL_BUILD_DIR)/%: $(TOOL_BUILD_DIR)/%.bin | $(TOOL_BUILD_DIR)
	@ $(RM) $@
	@ ln -s $(abspath $<) $@

$(TOOL_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(EXAMPLE_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(LINKFLAGS) $(LDFLAGS)
	@ echo

$(LAYER_BUILD_DIR)/%.o: src/$(PROJECT)/layers/%.cpp $(HXX_SRCS) \
		| $(LAYER_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(PROTO_BUILD_DIR)/%.pb.o: $(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_GEN_HEADER) \
		| $(PROTO_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(UTIL_BUILD_DIR)/%.o: src/$(PROJECT)/util/%.cpp $(HXX_SRCS) | $(UTIL_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(GTEST_OBJ): $(GTEST_SRC) | $(GTEST_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(LAYER_BUILD_DIR)/%.cuo: src/$(PROJECT)/layers/%.cu $(HXX_SRCS) \
		| $(LAYER_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(UTIL_BUILD_DIR)/%.cuo: src/$(PROJECT)/util/%.cu | $(UTIL_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(TOOL_BUILD_DIR)/%.o: tools/%.cpp $(PROTO_GEN_HEADER) | $(TOOL_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(EXAMPLE_BUILD_DIR)/%.o: examples/%.cpp $(PROTO_GEN_HEADER) \
		| $(EXAMPLE_BUILD_DIRS)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

$(BUILD_DIR)/src/$(PROJECT)/%.o: src/$(PROJECT)/%.cpp $(HXX_SRCS)
	$(CXX) $< $(CXXFLAGS) -c -o $@ 2> $@.$(WARNS_EXT) \
		|| (cat $@.$(WARNS_EXT); exit 1)
	@ cat $@.$(WARNS_EXT)
	@ echo

proto: $(PROTO_GEN_CC) $(PROTO_GEN_HEADER)

$(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_BUILD_DIR)/%.pb.h : \
		$(PROTO_SRC_DIR)/%.proto | $(PROTO_BUILD_DIR)
	protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<
	@ echo

$(PY_PROTO_BUILD_DIR)/%_pb2.py : $(PROTO_SRC_DIR)/%.proto \
		$(PY_PROTO_INIT) | $(PY_PROTO_BUILD_DIR)
	protoc --proto_path=$(PROTO_SRC_DIR) --python_out=$(PY_PROTO_BUILD_DIR) $<
	@ echo

$(PY_PROTO_INIT): | $(PY_PROTO_BUILD_DIR)
	touch $(PY_PROTO_INIT)

clean:
	@- $(RM) -rf $(ALL_BUILD_DIRS)
	@- $(RM) -rf $(OTHER_BUILD_DIR)
	@- $(RM) -rf $(BUILD_DIR_LINK)
	@- $(RM) -rf $(DISTRIBUTE_DIR)
	@- $(RM) $(PY$(PROJECT)_SO)
	@- $(RM) $(MAT$(PROJECT)_SO)

supercleanfiles:
	$(eval SUPERCLEAN_FILES := $(strip \
			$(foreach ext,$(SUPERCLEAN_EXTS), $(shell find . -name '*$(ext)' \
			-not -path './data/*'))))

supercleanlist: supercleanfiles
	@ \
	if [ -z "$(SUPERCLEAN_FILES)" ]; then \
		echo "No generated files found."; \
	else \
		echo $(SUPERCLEAN_FILES) | tr ' ' '\n'; \
	fi

superclean: clean supercleanfiles
	@ \
	if [ -z "$(SUPERCLEAN_FILES)" ]; then \
		echo "No generated files found."; \
	else \
		echo "Deleting the following generated files:"; \
		echo $(SUPERCLEAN_FILES) | tr ' ' '\n'; \
		$(RM) $(SUPERCLEAN_FILES); \
	fi

$(DIST_ALIASES): $(DISTRIBUTE_DIR)

$(DISTRIBUTE_DIR): all py $(HXX_SRCS) | $(DISTRIBUTE_SUBDIRS)
	# add include
	cp -r include $(DISTRIBUTE_DIR)/
	mkdir -p $(DISTRIBUTE_DIR)/include/caffe/proto
	cp $(PROTO_GEN_HEADER_SRCS) $(DISTRIBUTE_DIR)/include/caffe/proto
	# add tool and example binaries
	cp $(TOOL_BINS) $(DISTRIBUTE_DIR)/bin
	cp $(EXAMPLE_BINS) $(DISTRIBUTE_DIR)/bin
	# add libraries
	cp $(NAME) $(DISTRIBUTE_DIR)/lib
	cp $(STATIC_NAME) $(DISTRIBUTE_DIR)/lib
	# add python - it's not the standard way, indeed...
	cp -r python $(DISTRIBUTE_DIR)/python
