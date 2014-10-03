project( Examples )

file(GLOB_RECURSE EXAMPLES_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(source ${EXAMPLES_SOURCES})
    #    get file name
    get_filename_component(name ${source} NAME_WE)  
    
    #get folder name
    get_filename_component(path ${source} PATH)      
    get_filename_component(folder ${path} NAME_WE)      
    
    add_executable(${name} ${source})
    target_link_libraries(${name} caffe)
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${folder})
endforeach(source)
