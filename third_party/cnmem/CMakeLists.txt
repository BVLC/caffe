# CMakeLists to build the cnmem library.
cmake_minimum_required(VERSION 2.8.8)
project(cnmem)

# We need CUDA to build that library.
find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Rules to build the cnmem library.
include_directories(include)
add_definitions(-DCNMEM_DLLEXPORT)
add_library(cnmem SHARED src/cnmem.cpp)
set_target_properties(cnmem PROPERTIES VERSION 1.0.0 SOVERSION 1)
target_link_libraries(cnmem LINK_PUBLIC ${CUDA_LIBRARIES})
install(TARGETS cnmem RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES include/cnmem.h DESTINATION include)

# Add the tests.
if(WITH_TESTS)

  # Get Google tests.
  find_package(GTest QUIET REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})
  
  # Build the executable.
  add_executable(cnmem_tests tests/cnmem_tests.cpp)
  if(MSVC)
    if(MSVC_VERSION GREATER 1700) # Visual Studio 11 or more.
      add_definitions(-DUSE_CPP_11)
    endif(MSVC_VERSION GREATER 1700)
  endif(MSVC)
  if(CMAKE_COMPILER_IS_GNUCC)
    add_definitions(-std=c++11 -DUSE_CPP_11)
  endif(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(cnmem_tests LINK_PUBLIC cnmem ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} -lpthread)
  install(TARGETS cnmem_tests RUNTIME DESTINATION bin)
  
  # On Windows, we copy the Google test DLL to the bin folder.
  if(MSVC)
    get_filename_component(gtest_dll_path ${GTEST_LIBRARIES} DIRECTORY)
    install(FILES ${gtest_dll_path}/gtest.dll DESTINATION bin)
  endif(MSVC)

endif(WITH_TESTS)

