
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

  file(READ mkldnn.commit MKLDNN_COMMIT)

  include(ProcessorCount)
  ProcessorCount(NCORE)
  if(NOT NCORE EQUAL 0)
      set(CTEST_BUILD_FLAGS -j${NCORE})
      set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${NCORE})
  endif()

  if(MSVC)
      set(MKLDNN_CMAKE_GENERATOR "Visual Studio 15 2017 Win64")
      set(MKLDNN_INSTALL_CMD msbuild /p:Configuration=Release /m INSTALL.vcxproj)
  else()
      set(MKLDNN_INSTALL_CMD make install -j${NCORE})
  endif()

  ExternalProject_add(MKLDNN_Build
                      SOURCE_DIR ${MKLDNN_SOURCE_DIR}
                      CMAKE_GENERATOR ${MKLDNN_CMAKE_GENERATOR}
                      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${MKLDNN_INSTALL_DIR} -DMKLROOT=${MKL_ROOT_DIR} -DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}
#--Download step
                      GIT_REPOSITORY https://github.com/intel/mkl-dnn.git
                      GIT_TAG ${MKLDNN_COMMIT}
#--Build step
                      BINARY_DIR ${MKLDNN_BUILD_DIR}
                      BUILD_COMMAND cmake ${MKLDNN_SOURCE_DIR}
#--Install step
                      INSTALL_DIR ${MKLDNN_INSTALL_DIR}
                      INSTALL_COMMAND ${MKLDNN_INSTALL_CMD}
                      LOG_CONFIGURE 1
                      LOG_BUILD 1
                      LOG_INSTALL 1
                      ) 

  set(MKLDNN_INCLUDE_DIR ${MKLDNN_INSTALL_DIR}/include CACHE PATH "Include files for MKLDNN")
  set(MKLDNN_LIB_DIR ${MKLDNN_INSTALL_DIR}/${LIBDIR})
  add_library(mkldnn SHARED IMPORTED ${MKLDNN_INSTALL_DIR})
  set_property(TARGET mkldnn PROPERTY IMPORTED_LOCATION ${MKLDNN_LIB_DIR}/${MKLDNN_LIBRARIES_NAME})
  add_dependencies(mkldnn MKLDNN_Build)
endfunction(Download_MKLDNN)
