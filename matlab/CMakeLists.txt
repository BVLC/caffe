# Builds Matlab (or Octave) interface. In case of Matlab caffe must be
# compiled as shared library. Octave can link static or shared caffe library
# To install octave run: sudo apt-get install liboctave-dev

if(NOT BUILD_matlab)
  return()
endif()

if(HAVE_MATLAB AND Octave_compiler)
  set(build_using ${Matlab_build_mex_using})
elseif(HAVE_MATLAB AND NOT Octave_compiler)
  set(build_using "Matlab")
elseif(NOT HAVE_MATLAB AND Octave_compiler)
  set(build_using "Octave")
else()
  return()
endif()

if(NOT MSVC AND NOT BUILD_SHARED_LIBS AND build_using MATCHES Matlab)
  message(FATAL_ERROR "Matlab MEX interface (with default mex options file) can only be built if caffe is compiled as shared library on UNIX systems. Please enable 'BUILD_SHARED_LIBS' in CMake. Alternatively you can switch to Octave compiler.")
endif()

if(NOT MSVC)
# helper function to set proper mex file extension
  function(caffe_fetch_and_set_proper_mexext mexfile_variable)
    execute_process(COMMAND ${Matlab_mexext} OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE res OUTPUT_VARIABLE ext)
    if(res MATCHES 0)
      get_filename_component(folder  ${${mexfile_variable}} PATH)
      get_filename_component(name_we ${${mexfile_variable}} NAME_WE)
      set(${mexfile_variable} ${folder}/${name_we}.${ext} PARENT_SCOPE)
    endif()
  endfunction()
endif()

# global settings
file(GLOB Matlab_srcs +caffe/private/caffe_.cpp)
if(MSVC)
  set(Matlab_caffe_mex ${PROJECT_SOURCE_DIR}/matlab/+caffe/private/caffe_.mexw64)
else()
  set(Matlab_caffe_mex ${PROJECT_SOURCE_DIR}/matlab/+caffe/private/caffe_.mex)
endif()

caffe_get_current_cflags(cflags)
if(NOT MSVC)
  caffe_parse_linker_libs(Caffe_LINKER_LIBS folders libflags macos_frameworks)
endif()
set(folders $<TARGET_LINKER_FILE_DIR:caffe> ${folders})

# prepare linker flag lists
string(REPLACE ";" ";-L" link_folders "-L${folders}")
string(REPLACE ";" ":"  rpath_folders   "${folders}")

if(build_using MATCHES "Matlab")
  if(MSVC)
    matlab_add_mex(NAME matlab
                   SRC ${Matlab_srcs} # maybe you need to add some other sources
                   OUTPUT_NAME caffe_ # change the output name to _caffe.mexw64
                   LINK_TO caffe # cmake will take care of forwarding the correct transitive library dependencies to your mex file
                   )
    # output the target in the source tree as in the original version.
    set_target_properties(matlab PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/matlab/+caffe/private
                          )
    if(COPY_PREREQUISITES)
      caffe_copy_prerequisites(matlab DESTINATION ${PROJECT_SOURCE_DIR}/matlab/+caffe/private USE_HARD_LINKS)
    endif()
  else()
    set(libflags -lcaffe${Caffe_POSTFIX} ${libflags}) # Matlab R2014a complans for -Wl,--whole-archive

    caffe_fetch_and_set_proper_mexext(Matlab_caffe_mex)
    add_custom_command(OUTPUT ${Matlab_caffe_mex} COMMAND ${Matlab_mex}
        ARGS -output ${Matlab_caffe_mex} ${Matlab_srcs} ${cflags} ${link_folders} ${libflags}
        DEPENDS caffe COMMENT "Building Matlab interface: ${Matlab_caffe_mex}" VERBATIM)
    add_custom_target(matlab ALL DEPENDS ${Matlab_caffe_mex} SOURCES ${Matlab_srcs})
  endif()
elseif(build_using MATCHES "Octave")

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(libflags -Wl,-force_load,$<TARGET_LINKER_FILE:caffe> ${libflags})
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(libflags -Wl,--whole-archive -lcaffe${Caffe_POSTFIX} -Wl,--no-whole-archive ${libflags})
  endif()

  add_custom_command(OUTPUT ${Matlab_caffe_mex} COMMAND ${Octave_compiler}
      ARGS --mex -o ${Matlab_caffe_mex} ${Matlab_srcs} ${cflags} ${link_folders} ${libflags} -Wl,-rpath,${rpath_folders}
      DEPENDS caffe COMMENT "Building Octave interface: ${Matlab_caffe_mex}" VERBATIM)

  add_custom_target(octave ALL DEPENDS ${Matlab_caffe_mex} SOURCES ${Matlab_srcs})
endif()

# ---[ Install
if(MSVC)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/matlab DESTINATION .
          PATTERN CMakeLists.txt EXCLUDE
          PATTERN .gitignore EXCLUDE)
else()
  file(GLOB mfiles caffe/*.m)
  install(FILES ${mfiles} ${Matlab_caffe_mex} DESTINATION matlab)
endif()

if(MSVC AND INSTALL_PREREQUISITES)
  caffe_install_prerequisites(matlab DESTINATION matlab/+caffe/private)
endif()

