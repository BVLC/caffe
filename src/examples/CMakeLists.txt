project( Examples )

#    Find all source files
file(GLOB_RECURSE Examples_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

#    Build each source file independently
foreach(source ${Examples_SOURCES})
    get_filename_component(name ${source} NAME_WE)  
    add_executable(${name} ${source})
    target_link_libraries(${name} caffe)
endforeach(source)