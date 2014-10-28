project( Tools )

#    Find all source files
file(GLOB_RECURSE TOOLS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

#    Build each source file independently
foreach(source ${TOOLS_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name}.bin ${source})
    set_target_properties(${name}.bin PROPERTIES OUTPUT_NAME ${name})
    target_link_libraries(${name}.bin caffe)

###    Install    #################################################################################

    install(TARGETS ${name}.bin DESTINATION tools)


endforeach(source)

