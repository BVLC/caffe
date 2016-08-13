# This is a specialized version of caffe/halide/CMakeLists.txt
# make sure to update this file to if you improve this one

# ---[ Adding generator target
file(GLOB gen_srcs generator/*.cpp)
set_source_files_properties(${gen_srcs} PROPERTIES COMPILE_FLAGS "-std=c++11 -g -fno-rtti")

link_directories(${HALIDE_BIN_DIR})

add_executable(halide_object_generator_test ${gen_srcs})

target_include_directories(halide_object_generator_test PUBLIC ${HALIDE_INCLUDE_DIR})
target_link_libraries(halide_object_generator_test Halide pthread dl  pthread z)
caffe_default_properties(halide_object_generator_test)
caffe_set_runtime_directory(halide_object_generator_test "${PROJECT_BINARY_DIR}/halide")

# compile the wrapper, specify this first because we want to have their target handles
# so that we can add all objects to it ( which we will specify next
# this is done so that we can have a many objects to one wrapper mapping

#file(GLOB wrappers wrappers/plip_wrapper.cpp)
#FOREACH(wrapperCppPath ${wrappers})
set( wrapperCppPath wrappers/plip_wrapper.cpp)

# Generate output file name
get_filename_component(halide_wrapper ${wrapperCppPath} NAME_WE)
#MESSAGE(STATUS "Process file: ${wrapperCppPath} -> halide_wrapper")
set_source_files_properties(${wrapperCppPath} PROPERTIES COMPILE_FLAGS "-shared -g -std=c++11 -fPIC -fno-rtti")
add_library(halide_wrapper ${wrapperCppPath} )
target_include_directories(halide_wrapper PUBLIC ${HALIDE_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/halide)
target_link_libraries(halide_wrapper pthread dl pthread z glog gflags boost_system-mt caffe Halide)
caffe_default_properties(halide_wrapper)
caffe_set_runtime_directory(halide_wrapper "${PROJECT_BINARY_DIR}/halide")
#install(TARGETS halide_wrapper DESTINATION tests)
#caffe_set_runtime_directory(halide_wrapper "${PROJECT_BINARY_DIR}/test")
set_target_properties(halide_wrapper PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test")

#ENDFOREACH(wrapperCppPath)



#file(GLOB gens generator/*.gen)
#FOREACH(infilePath ${gens})
set( infilePath generator/plip.gen )

# Generate output file name
get_filename_component(in_obj ${infilePath} NAME_WE)
#MESSAGE(STATUS "Process file: ${gens} -> ${in_obj}")
SET(outfile "${PROJECT_BINARY_DIR}/halide/${in_obj}")
#MESSAGE(WARNING "Output file: ${outfile}.o")
add_custom_target(halide_generator_run_test
  #OUTPUT "${outfile}"
  COMMAND halide_object_generator_test -g ${in_obj} -o . target=cuda
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/halide
  COMMENT "run generated halide_object_generator_test in ${PROJECT_BINARY_DIR}/halide"
  SOURCES ${halide_object_SOURCES}
)
#FOREACH(wrapperCppPath ${wrappers})
# Generate output file name
get_filename_component(wrapper_obj ${wrapperCppPath} NAME_WE)
add_dependencies(halide_wrapper halide_generator_run_test)
#ENDFOREACH(wrapperCppPath)
#ENDFOREACH(infilePath)
