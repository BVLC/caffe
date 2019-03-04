
function(Download_MKLDNN)
  set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
  set(MKLDNN_DIR ${EXTERNAL_DIR}/mkldnn)
  set(MKLDNN_SOURCE_DIR ${MKLDNN_DIR}/src)
  set(MKLDNN_BUILD_DIR ${MKLDNN_DIR}/build)
  set(MKLDNN_INSTALL_DIR ${MKLDNN_DIR}/install CACHE PATH "Installation path of MKLDNN")
  # Enable MKLDNN intel compiler static build
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    if(ICC_STATIC_BUILD)
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-intel")
    endif()
  endif()
  execute_process(COMMAND cat mkldnn.commit
  		  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		  OUTPUT_VARIABLE MKLDNN_COMMIT)

  include(ProcessorCount)
  ProcessorCount(NCORE)
  if(NOT NCORE EQUAL 0)
      set(CTEST_BUILD_FLAGS -j${NCORE})
      set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${NCORE})
  endif()
 
  ExternalProject_add(MKLDNN_Build
                      SOURCE_DIR ${MKLDNN_SOURCE_DIR}
                      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR} -DMKLROOT=${MKL_ROOT_DIR} -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
#--Download step
                      GIT_REPOSITORY https://github.com/01org/mkl-dnn.git
                      GIT_TAG ${MKLDNN_COMMIT}
#--Build step
                      BINARY_DIR ${MKLDNN_BUILD_DIR}
                      BUILD_COMMAND cmake ${MKLDNN_SOURCE_DIR} 
#--Install step
                      INSTALL_DIR ${MKLDNN_INSTALL_DIR}
                      INSTALL_COMMAND make install -j${NCORE}
                      LOG_CONFIGURE 1
                      LOG_BUILD 1
                      LOG_INSTALL 1
                      ) 

  set(MKLDNN_INCLUDE_DIR ${MKLDNN_INSTALL_DIR}/include CACHE PATH "Include files for MKLDNN")
  set(MKLDNN_LIB_DIR ${MKLDNN_INSTALL_DIR}/lib64)
  add_library(mkldnn SHARED IMPORTED ${MKLDNN_INSTALL_DIR})
  set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/libmkldnn.so)
  add_dependencies(mkldnn MKLDNN_Build)
endfunction(Download_MKLDNN)
