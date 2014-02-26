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
CXX_SRCS := $(shell find src/$(PROJECT) ! -name "test_*.cpp" -name "*.cpp")
# HXX_SRCS are the header files
HXX_SRCS := $(shell find include/$(PROJECT) ! -name "*.hpp")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find src/$(PROJECT) -name "*.cu")
# TEST_SRCS are the test source files
TEST_SRCS := $(shell find src/$(PROJECT) -name "test_*.cpp")
GTEST_SRC := src/gtest/gtest-all.cpp
# TEST_HDRS are the test header files
TEST_HDRS := $(shell find src/$(PROJECT) -name "test_*.hpp")
# TOOL_SRCS are the source files for the tool binaries
TOOL_SRCS := $(shell find tools -name "*.cpp")
# EXAMPLE_SRCS are the source files for the example binaries
EXAMPLE_SRCS := $(shell find examples -name "*.cpp")
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRCS := $(wildcard src/$(PROJECT)/proto/*.proto)
# NONGEN_CXX_SRCS includes all source/header files except those generated
# automatically (e.g., by proto).
NONGEN_CXX_SRCS := $(shell find \
	src/$(PROJECT) \
	include/$(PROJECT) \
	python/$(PROJECT) \
	matlab/$(PROJECT) \
	examples \
	tools \
	-regex ".*\.\(cpp\|hpp\|cu\|cuh\)")
LINT_REPORT := $(BUILD_DIR)/cpp_lint.log
FAILED_LINT_REPORT := $(BUILD_DIR)/cpp_lint.error_log
# PY$(PROJECT)_SRC is the python wrapper for $(PROJECT)
PY$(PROJECT)_SRC := python/$(PROJECT)/py$(PROJECT).cpp
PY$(PROJECT)_SO := python/$(PROJECT)/py$(PROJECT).so
# MAT$(PROJECT)_SRC is the matlab wrapper for $(PROJECT)
MAT$(PROJECT)_SRC := matlab/$(PROJECT)/mat$(PROJECT).cpp
MAT$(PROJECT)_SO := matlab/$(PROJECT)/$(PROJECT)

##############################
# Derive generated files
##############################
# The generated files for protocol buffers
PROTO_GEN_HEADER := ${PROTO_SRCS:.proto=.pb.h}
PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}
PROTO_GEN_PY := ${PROTO_SRCS:.proto=_pb2.py}
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the tool, example, and test objects.
CXX_OBJS := $(addprefix $(BUILD_DIR)/, ${CXX_SRCS:.cpp=.o})
CU_OBJS := $(addprefix $(BUILD_DIR)/, ${CU_SRCS:.cu=.cuo})
PROTO_OBJS := $(addprefix $(BUILD_DIR)/, ${PROTO_GEN_CC:.cc=.o})
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)
# tool, example, and test objects
TOOL_OBJS := $(addprefix $(BUILD_DIR)/, ${TOOL_SRCS:.cpp=.o})
EXAMPLE_OBJS := $(addprefix $(BUILD_DIR)/, ${EXAMPLE_SRCS:.cpp=.o})
TEST_OBJS := $(addprefix $(BUILD_DIR)/, ${TEST_SRCS:.cpp=.o})
GTEST_OBJ := $(addprefix $(BUILD_DIR)/, ${GTEST_SRC:.cpp=.o})
# tool, example, and test bins
TOOL_BINS := ${TOOL_OBJS:.o=.bin}
EXAMPLE_BINS := ${EXAMPLE_OBJS:.o=.bin}
TEST_BINS := ${TEST_OBJS:.o=.testbin}

##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64 $(CUDA_DIR)/lib
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

INCLUDE_DIRS += ./src ./include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS += $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cudart cublas curand \
	mkl_rt \
	pthread \
	glog protobuf leveldb \
	snappy \
	boost_system \
	hdf5_hl hdf5 \
	opencv_core opencv_highgui opencv_imgproc
PYTHON_LIBRARIES := boost_python python2.7
WARNINGS := -Wall

COMMON_FLAGS := -DNDEBUG -O2 $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
NVCCFLAGS := -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
		$(foreach library,$(LIBRARIES),-l$(library))
PYTHON_LDFLAGS := $(LDFLAGS) $(foreach library,$(PYTHON_LIBRARIES),-l$(library))


##############################
# Define build targets
##############################
.PHONY: all init test clean linecount lint tools examples py mat distribute py$(PROJECT) mat$(PROJECT) proto

all: init $(NAME) $(STATIC_NAME) tools examples
	@echo $(CXX_OBJS)

init:
	@ mkdir -p $(foreach obj,$(OBJS),$(dir $(obj)))
	@ mkdir -p $(foreach obj,$(TOOL_OBJS),$(dir $(obj)))
	@ mkdir -p $(foreach obj,$(EXAMPLE_OBJS),$(dir $(obj)))
	@ mkdir -p $(foreach obj,$(TEST_OBJS),$(dir $(obj)))
	@ mkdir -p $(foreach obj,$(GTEST_OBJ),$(dir $(obj)))

linecount: clean
	cloc --read-lang-def=$(PROJECT).cloc src/$(PROJECT)/

lint: $(LINT_REPORT)

$(LINT_REPORT): $(NONGEN_CXX_SRCS)
	@ mkdir -p $(BUILD_DIR)
	@ (python ./scripts/cpp_lint.py $(NONGEN_CXX_SRCS) > $(LINT_REPORT) 2>&1 \
		&& (rm -f $(FAILED_LINT_REPORT); echo "No linter errors!")) || ( \
			mv $(LINT_REPORT) $(FAILED_LINT_REPORT); \
			grep -v "^Done processing " $(FAILED_LINT_REPORT); \
			echo "Found 1 or more linter errors; see log at $(FAILED_LINT_REPORT)"; \
			exit 1)

test: init $(TEST_BINS)

tools: init $(TOOL_BINS)

examples: init $(EXAMPLE_BINS)

py$(PROJECT): py

py: init $(STATIC_NAME) $(PY$(PROJECT)_SRC) $(PROTO_GEN_PY)
	$(CXX) -shared -o $(PY$(PROJECT)_SO) $(PY$(PROJECT)_SRC) \
		$(STATIC_NAME) $(CXXFLAGS) $(PYTHON_LDFLAGS)
	@echo

mat$(PROJECT): mat

mat: init $(STATIC_NAME) $(MAT$(PROJECT)_SRC)
	$(MATLAB_DIR)/bin/mex $(MAT$(PROJECT)_SRC) $(STATIC_NAME) \
		CXXFLAGS="\$$CXXFLAGS $(CXXFLAGS) $(WARNINGS)" \
		CXXLIBS="\$$CXXLIBS $(LDFLAGS)" \
		-o $(MAT$(PROJECT)_SO)
	@echo

$(NAME): init $(PROTO_OBJS) $(OBJS)
	$(CXX) -shared -o $(NAME) $(OBJS) $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@echo

$(STATIC_NAME): init $(PROTO_OBJS) $(OBJS)
	ar rcs $(STATIC_NAME) $(PROTO_OBJS) $(OBJS)
	@echo

runtest: test
	for testbin in $(TEST_BINS); do $$testbin $(TEST_GPUID); done

$(TEST_BINS): %.testbin : %.o $(GTEST_OBJ) $(STATIC_NAME) $(TEST_HDRS)
	$(CXX) $< $(GTEST_OBJ) $(STATIC_NAME) -o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)

$(TOOL_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@echo

$(EXAMPLE_BINS): %.bin : %.o $(STATIC_NAME)
	$(CXX) $< $(STATIC_NAME) -o $@ $(CXXFLAGS) $(LDFLAGS) $(WARNINGS)
	@echo

$(OBJS): $(PROTO_GEN_CC) $(HXX_SRCS)

$(BUILD_DIR)/src/$(PROJECT)/%.o: src/$(PROJECT)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

$(BUILD_DIR)/src/$(PROJECT)/layers/%.o: src/$(PROJECT)/layers/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

$(BUILD_DIR)/src/$(PROJECT)/proto/%.o: src/$(PROJECT)/proto/%.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

$(BUILD_DIR)/src/$(PROJECT)/test/%.o: src/test/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

$(BUILD_DIR)/src/$(PROJECT)/util/%.o: src/$(PROJECT)/util/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

$(BUILD_DIR)/src/gtest/%.o: src/gtest/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@
	@echo

$(BUILD_DIR)/src/$(PROJECT)/layers/%.cuo: src/$(PROJECT)/layers/%.cu
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
	@echo

$(BUILD_DIR)/src/$(PROJECT)/util/%.cuo: src/$(PROJECT)/util/%.cu
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
	@echo

$(BUILD_DIR)/tools/%.o: tools/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@echo

$(BUILD_DIR)/examples/%.o: examples/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@echo

$(PROTO_GEN_PY): $(PROTO_SRCS)
	protoc --proto_path=src --python_out=python $(PROTO_SRCS)
	@echo

proto: $(PROTO_GEN_CC)

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc --proto_path=src --cpp_out=src $(PROTO_SRCS)
	mkdir -p include/$(PROJECT)/proto
	cp $(PROTO_GEN_HEADER) include/$(PROJECT)/proto/
	@echo

clean:
	@- $(RM) $(NAME) $(STATIC_NAME)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)
	@- $(RM) include/$(PROJECT)/proto/$(PROJECT).pb.h
	@- $(RM) python/$(PROJECT)/proto/$(PROJECT)_pb2.py
	@- $(RM) python/$(PROJECT)/*.so
	@- $(RM) -rf $(BUILD_DIR)
	@- $(RM) -rf $(DISTRIBUTE_DIR)

distribute: all
	mkdir $(DISTRIBUTE_DIR)
	# add include
	cp -r include $(DISTRIBUTE_DIR)/
	# add tool and example binaries
	mkdir $(DISTRIBUTE_DIR)/bin
	cp $(TOOL_BINS) $(DISTRIBUTE_DIR)/bin
	cp $(EXAMPLE_BINS) $(DISTRIBUTE_DIR)/bin
	# add libraries
	mkdir $(DISTRIBUTE_DIR)/lib
	cp $(NAME) $(DISTRIBUTE_DIR)/lib
	cp $(STATIC_NAME) $(DISTRIBUTE_DIR)/lib
	# add python - it's not the standard way, indeed...
	cp -r python $(DISTRIBUTE_DIR)/python
