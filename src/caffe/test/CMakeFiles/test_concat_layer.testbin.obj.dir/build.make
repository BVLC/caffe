# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xkcd/Documents/external_code/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xkcd/Documents/external_code/caffe

# Include any dependencies generated for this target.
include src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/depend.make

# Include the progress variables for this target.
include src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/progress.make

# Include the compile flags for this target's objects.
include src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/flags.make

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o: src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/flags.make
src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o: src/caffe/test/test_concat_layer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/xkcd/Documents/external_code/caffe/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o"
	cd /home/xkcd/Documents/external_code/caffe/src/caffe/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o -c /home/xkcd/Documents/external_code/caffe/src/caffe/test/test_concat_layer.cpp

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.i"
	cd /home/xkcd/Documents/external_code/caffe/src/caffe/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/xkcd/Documents/external_code/caffe/src/caffe/test/test_concat_layer.cpp > CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.i

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.s"
	cd /home/xkcd/Documents/external_code/caffe/src/caffe/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/xkcd/Documents/external_code/caffe/src/caffe/test/test_concat_layer.cpp -o CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.s

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.requires:
.PHONY : src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.requires

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.provides: src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.requires
	$(MAKE) -f src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/build.make src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.provides.build
.PHONY : src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.provides

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.provides.build: src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o

test_concat_layer.testbin.obj: src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o
test_concat_layer.testbin.obj: src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/build.make
.PHONY : test_concat_layer.testbin.obj

# Rule to build all files generated by this target.
src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/build: test_concat_layer.testbin.obj
.PHONY : src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/build

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/requires: src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/test_concat_layer.cpp.o.requires
.PHONY : src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/requires

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/clean:
	cd /home/xkcd/Documents/external_code/caffe/src/caffe/test && $(CMAKE_COMMAND) -P CMakeFiles/test_concat_layer.testbin.obj.dir/cmake_clean.cmake
.PHONY : src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/clean

src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/depend:
	cd /home/xkcd/Documents/external_code/caffe && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xkcd/Documents/external_code/caffe /home/xkcd/Documents/external_code/caffe/src/caffe/test /home/xkcd/Documents/external_code/caffe /home/xkcd/Documents/external_code/caffe/src/caffe/test /home/xkcd/Documents/external_code/caffe/src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/caffe/test/CMakeFiles/test_concat_layer.testbin.obj.dir/depend

