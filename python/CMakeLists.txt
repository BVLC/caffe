project( Python )

#    Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
# TODO: Numpy should be found with a find_package command
include_directories(/usr/local/anaconda/lib/python2.7/site-packages/numpy/core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/caffe)

#    Boost.Python
find_package(Boost 1.46 COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB_RECURSE Python_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/caffe/*.cpp)

add_library(pycaffe SHARED ${Python_SOURCES})
set_target_properties(pycaffe PROPERTIES OUTPUT_NAME _caffe PREFIX "")
target_link_libraries(pycaffe caffe ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

###    Install    #################################################################################

install(DIRECTORY caffe DESTINATION python)
install(FILES requirements.txt DESTINATION python)
install(TARGETS pycaffe DESTINATION python/caffe)