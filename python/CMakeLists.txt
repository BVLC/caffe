if(NOT HAVE_PYTHON)
  message(STATUS "Python interface is disabled or not all required dependecies found. Building without it...")
  return()
endif()

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
file(GLOB_RECURSE python_srcs ${CMAKE_SOURCE_DIR}/python/*.cpp)

add_library(pycaffe SHARED ${python_srcs})
target_link_libraries(pycaffe ${Caffe_STATIC_LINK} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
set_target_properties(pycaffe PROPERTIES PREFIX "" OUTPUT_NAME "_caffe")
caffe_default_properties(pycaffe)

if(UNIX OR APPLE)
    set(__linkname "${CMAKE_SOURCE_DIR}/python/_caffe.so")
    add_custom_command(TARGET pycaffe POST_BUILD
                       COMMAND ln -sf $<TARGET_LINKER_FILE:caffe> "${CMAKE_SOURCE_DIR}/python/_caffe.so"
                       COMMENT "Creating simlink ${__linkname} -> ${CMAKE_BINARY_DIR}/lib/_caffe${CAffe_POSTFIX}.so")
endif()

# ---[ Install
file(GLOB files *.py requirements.txt)
install(FILES  ${files} DESTINATION python)
install(DIRECTORY caffe DESTINATION python)
install(TARGETS pycaffe DESTINATION python/caffe)


