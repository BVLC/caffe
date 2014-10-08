project( Python )

#    Python
find_package(PythonLibs 2.7 REQUIRED)

#   Numpy
find_package(NumPy REQUIRED)

#    Boost.Python
find_package(Boost 1.46 COMPONENTS python REQUIRED)



#In case you have both python2 and python3 installed the quickest way to
#compile pycaffe with cmake is to replace the following hardcoded paths.
#Althernativley the Find* scripts could be rewritten to support choice of
#of python version.
#if(${PYTHONLIBS_VERSION_STRING} MATCHES "^[3-9]+\\.[0-9]+(\\.[0-9]+.*)?$") 
#    
#    set( PYTHON_INCLUDE_DIRS "/usr/include/python2.7")
#    set( PYTHON_LIBRARIES    "/usr/lib64/libpython2.7.so")
#    set( NUMPY_INCLUDE_DIRS  "/usr/lib64/python2.7/site-packages/numpy/core/include/")
#    set( PYTHON_LIBRARIES    "/usr/lib64/python2.7/site-packages/numpy/lib/")
#    set(Boost_LIBRARIES      "/usr/lib64/libboost_python-2.7-mt.so")
#
#    message( "Warning: cmake found python3 by default, switching to hardcoded paths")
#
#    message( "PYTHON_INCLUDE_DIRS =/usr/include/python2.7")
#    message( "PYTHON_LIBRARIES    =/usr/lib64/libpython2.7.so")
#    message( "NUMPY_INCLUDE_DIRS  =/usr/lib64/python2.7/site-packages/numpy/core/include/")
#    message( "PYTHON_LIBRARIES    =/usr/lib64/python2.7/site-packages/numpy/lib/")
#    message( "Boost_LIBRARIES     =/usr/lib64/libboost_python-2.7-mt.so")
#endif()


include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})

file(GLOB_RECURSE Python_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(pycaffe SHARED ${Python_SOURCES})

add_dependencies(pycaffe protoPy)

target_link_libraries(pycaffe ${CAFFE_STATIC_LINK} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

set_target_properties(pycaffe PROPERTIES PREFIX "")
set_target_properties(pycaffe PROPERTIES OUTPUT_NAME "_caffe")

###    Install    #############################################################

install(DIRECTORY caffe DESTINATION python)
install(FILES requirements.txt DESTINATION python)

#This installs a library named "libpycaffe.so"
install(TARGETS pycaffe DESTINATION python/caffe)


