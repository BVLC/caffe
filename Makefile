# The makefile for caffe. Pretty hacky.
PROJECT := caffe

CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

# The target static library and shared library name
LIB_BUILD_DIR := $(BUILD_DIR)/lib
NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).so
STATIC_NAME := $(LIB_BUILD_DIR)/lib$(PROJECT).a

##############################
# Get all source files
##############################
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cpp" -name "*.cpp")
# HXX_SRCS are the header files
HXX_SRCS := $(shell find include/$(PROJECT) -name "*.hpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/$(PROJECT) -name "*.cu")
# TEST_SRCS are the test source files
TEST_MAIN_SRC := src/$(PROJECT)/test/test_caffe_main.cpp
TEST_SRCS := $(shell find src/$(PROJECT) -name "test_*.cpp")
TEST_SRCS := $(filter-out $(TEST_MAIN_SRC), $(TEST_SRCS))
GTEST_SRC := src/gtest/gtest-all.cpp
# TEST_HDRS are the test header files
TEST_HDRS := $(shell find src/$(PROJECT) -name "test_*.hpp")
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
LINT_REPORT := $(BUILD_DIR)/cpp_lint.log
FAILED_LINT_REPORT := $(BUILD_DIR)/cpp_lint.error_log
# PY$(PROJECT)_SRC is the python wrapper for $(PROJECT)
PY$(PROJECT)_SRC := python/$(PROJECT)/_$(PROJECT).cpp
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
TEST_OBJS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cpp=.o})
GTEST_OBJ := $(addprefix $(BUILD_DIR)/, ${GTEST_SRC:.cpp=.o})
GTEST_BUILD_DIR := $(dir $(GTEST_OBJ))
EXAMPLE_OBJS := $(addprefix $(BUILD_DIR)/, ${EXAMPLE_SRCS:.cpp=.o})
EXAMPLE_BUILD_DIR := $(BUILD_DIR)/examples
EXAMPLE_BUILD_DIRS := $(EXAMPLE_BUILD_DIR)
EXAMPLE_BUILD_DIRS += $(foreach obj,$(EXAMPLE_OBJS),$(dir $(obj)))
# tool, example, and test bins
TOOL_BINS := ${TOOL_OBJS:.o=.bin}
EXAMPLE_BINS := ${EXAMPLE_OBJS:.o=.bin}
# Put the test binaries in build/test for convenience.
TEST_BIN_DIR := $(BUILD_DIR)/test
TEST_BINS := $(addsuffix .testbin,$(addprefix $(TEST_BIN_DIR)/, \
		$(foreach obj,$(TEST_OBJS),$(basename $(notdir $(obj))))))
TEST_ALL_BIN := $(TEST_BIN_DIR)/test_all.testbin

##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64 $(CUDA_DIR)/lib

INCLUDE_DIRS += $(BUILD_INCLUDE_DIR)
INCLUDE_DIRS += ./src ./include $(CUDA_INCLUDE_DIR)
LIBRARY_DIRS += $(CUDA_LIB_DIR)
LIBRARIES := cudart cublas curand \
	pthread \
	glog protobuf leveldb snappy \
	boost_system \
	hdf5_hl hdf5 \
	opencv_core opencv_highgui opencv_imgproc
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall

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
		$(PROTO_BUILD_DIR) $(PROTO_BUILD_INCLUDE_DIR) $(PY_PROTO_BUILD_DIR) \
		$(DISTRIBUTE_SUBDIRS))

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
	CXX := /usr/bin/g++
endif

# OS X:
# clang++ instead of g++
# libstdc++ instead of libc++ for CUDA compatibility on 10.9
ifeq ($(OSX), 1)
	CXX := /usr/bin/clang++
	ifneq ($(findstring 10.9, $(shell sw_vers -productVersion)),)
		CXXFLAGS += -stdlib=libstdc++
	endif
endif

# Debugging
DEBUG ?= 0
ifeq ($(DEBUG), 1)
	COMMON_FLAGS := -DDEBUG -g -O0
else
	COMMON_FLAGS := -DNDEBUG -O2
endif

# BLAS configuration (default = ATLAS)
BLAS ?= atlas
ifeq ($(BLAS), mkl)
	# MKL
	LIBRARIES += mkl_rt
	COMMON_FLAGS += -DUSE_MKL
	MKL_DIR = /opt/intel/mkl
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
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
NVCCFLAGS := -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
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
.PHONY: all test clean linecount lint tools examples $(DIST_ALIASES) \
	py mat py$(PROJECT) mat$(PROJECT) proto runtest \
	superclean supercleanlist supercleanfiles

all: $(NAME) $(STATIC_NAME) tools examples

linecount: clean
	cloc --read-lang-def=$(PROJECT).cloc src/$(PROJECT)/

lint: $(LINT_REPORT)

$(LINT_REPORT): $(NONGEN_CXX_SRCS) | $(BUILD_DIR)
	@ (python ./scripts/cpp_lint.py $(NONGEN_CXX_SRCS) > $(LINT_REPORT) 2>&1 \
		&& ($(RM) $(FAILED_LINT_REPORT); echo "No lint errors!")) || ( \
			mv $(LINT_REPORT) $(FAILED_LINT_REPORT); \
			grep -v "^Done processing " $(FAILED_LINT_REPORT); \
			echo "Found 1 or more lint errors; see log at $(FAILED_LINT_REPORT)"; \
			exit 1)

test: $(TEST_ALL_BIN) $(TEST_BINS)

tools: $(TOOL_BINS)

examples: $(EXAMPLE_BINS)

py$(PROJECT): py

py: $(PY$(PROJECT)_SO) $(PROTO_GEN_PY)

$(PY$(PROJECT)_SO): $(STATIC_NAME) $(PY$(PROJECT)_SRC)
	$(CXX) -shared -o $@ $(PY$(PROJECT)_SRC) \
		$(STATIC_NAME) $(CXXFLAGS) $(PYTHON_LDFLAGS)
	@ echo

mat$(PROJECT): mat

mat: $(MAT$(PROJECT)_SO)

$(MAT$(PROJECT)_SO): $(MAT$(PROJECT)_SRC) $(STATIC_NAME)
	@ if [ -z "$(MATLAB_DIR)" ]; then \
		echo "MATLAB_DIR must be specified in $(CONFIG_FILE)" \
			"to build mat$(PROJECT)."; \
		exit 1; \
	fi
	$(MATLAB_DIR)/bin/mex $(MAT$(PROJECT)_SRC) $(STATIC_NAME) \
			CXXFLAGS="\$$CXXFLAGS $(CXXFLAGS) $(WARNINGS)" \
			CXXLIBS="\$$CXXLIBS $(LDFLAGS)" -o $@
	@ echo

runtest: $(TEST_ALL_BIN)
	$(TEST_ALL_BIN) $(TEST_GPUID) --gtest_shuffle

$(ALL_BUILD_DIRS):
	@ mkdir -p $@

$(NAME): $(PROTO_OBJS) $(OBJS) | $(LIB_BUILD_DIR)
	$(CXX) -shared -o $@ $(OBJS) $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@ echo

$(STATIC_NAME): $(PROTO_OBJS) $(OBJS) | $(LIB_BUILD_DIR)
	ar rcs $@ $(PROTO_OBJS) $(OBJS)
	@ echo

$(TEST_BUILD_DIR)/%.o: src/$(PROJECT)/test/%.cpp $(HXX_SRCS) $(TEST_HDRS) \
		| $(TEST_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

$(TEST_ALL_BIN): $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $(TEST_OBJS) $(GTEST_OBJ) $(STATIC_NAME) \
		-o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@ echo

$(TEST_BIN_DIR)/%.testbin: $(TEST_BUILD_DIR)/%.o $(GTEST_OBJ) $(STATIC_NAME) \
		| $(TEST_BIN_DIR)
	$(CXX) $(TEST_MAIN_SRC) $< $(GTEST_OBJ) $(STATIC_NAME) \
		-o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@ echo

$(TOOL_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@ echo

$(EXAMPLE_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@ echo

$(LAYER_BUILD_DIR)/%.o: src/$(PROJECT)/layers/%.cpp $(HXX_SRCS) \
		| $(LAYER_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

$(PROTO_BUILD_DIR)/%.pb.o: $(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_GEN_HEADER) \
		| $(PROTO_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

$(UTIL_BUILD_DIR)/%.o: src/$(PROJECT)/util/%.cpp $(HXX_SRCS) | $(UTIL_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

$(GTEST_OBJ): $(GTEST_SRC) | $(GTEST_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

$(LAYER_BUILD_DIR)/%.cuo: src/$(PROJECT)/layers/%.cu $(HXX_SRCS) \
		| $(LAYER_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
	@ echo

$(UTIL_BUILD_DIR)/%.cuo: src/$(PROJECT)/util/%.cu | $(UTIL_BUILD_DIR)
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
	@ echo

$(TOOL_BUILD_DIR)/%.o: tools/%.cpp $(PROTO_GEN_HEADER) | $(TOOL_BUILD_DIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@ echo

$(EXAMPLE_BUILD_DIR)/%.o: examples/%.cpp $(PROTO_GEN_HEADER) \
		| $(EXAMPLE_BUILD_DIRS)
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@ echo

$(BUILD_DIR)/src/$(PROJECT)/%.o: src/$(PROJECT)/%.cpp $(HXX_SRCS)
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@ echo

proto: $(PROTO_GEN_CC) $(PROTO_GEN_HEADER)

$(PROTO_BUILD_DIR)/%.pb.cc $(PROTO_BUILD_DIR)/%.pb.h : \
		$(PROTO_SRC_DIR)/%.proto | $(PROTO_BUILD_DIR)
	protoc --proto_path=src --cpp_out=build/src $<
	@ echo

$(PY_PROTO_BUILD_DIR)/%_pb2.py : $(PROTO_SRC_DIR)/%.proto \
		$(PY_PROTO_INIT) | $(PY_PROTO_BUILD_DIR)
	protoc --proto_path=src --python_out=python $<
	@ echo

$(PY_PROTO_INIT): | $(PY_PROTO_BUILD_DIR)
	touch $(PY_PROTO_INIT)

clean:
	@- $(RM) -rf $(ALL_BUILD_DIRS)
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
	# add tool and example binaries
	cp $(TOOL_BINS) $(DISTRIBUTE_DIR)/bin
	cp $(EXAMPLE_BINS) $(DISTRIBUTE_DIR)/bin
	# add libraries
	cp $(NAME) $(DISTRIBUTE_DIR)/lib
	cp $(STATIC_NAME) $(DISTRIBUTE_DIR)/lib
	# add python - it's not the standard way, indeed...
	cp -r python $(DISTRIBUTE_DIR)/python
