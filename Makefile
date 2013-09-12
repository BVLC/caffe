#
# The following defines a variable named "NAME" with a value of "myprogram". By convention,
# a lowercase prefix (in this case "program") and an uppercased suffix (in this case "NAME"), separated
# by an underscore is used to name attributes for a common element. Think of this like
# using program.NAME, program.C_SRCS, etc. There are no structs in Make, so we use this convention
# to keep track of attributes that all belong to the same target or program.  
#
NAME := caffeine.so
C_SRCS := $(wildcard src/caffeine/*.c)
CXX_SRCS := $(wildcard src/caffeine/*.cpp)
C_OBJS := ${C_SRCS:.c=.o}
CXX_OBJS := ${CXX_SRCS:.cpp=.o}
OBJS := $(C_OBJS) $(CXX_OBJS)

CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib

INCLUDE_DIRS := $(CUDA_INCLUDE_DIR) src/
LIBRARY_DIRS := $(CUDA_LIB_DIR)
LIBRARIES := cudart cublas
WARNINGS := -Wall

CPPFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library)) -shared

LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(WARNINGS)

.PHONY: all clean distclean

all: $(NAME)

$(NAME): $(OBJS)
	$(LINK) $(OBJS) -o $(NAME)

clean:
	@- $(RM) $(NAME)
	@- $(RM) $(OBJS)

distclean: clean