project( Tools )

#    Find all source files
file(GLOB_RECURSE Tools_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

#    Build each source file independently
foreach(source ${Tools_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} caffe)

###    Install    #################################################################################

    install(TARGETS ${name} DESTINATION tools)


endforeach(source)

