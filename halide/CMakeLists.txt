# definition needed to include CMake generated files

if(BUILD_halide)

    add_subdirectory(test)

    # compile the wrapper, specify this first because we want to have their target handles
    # so that we can add all objects to it ( which we will specify next
    # this is done so that we can have a many objects to one wrapper mapping

    file(GLOB wrappers wrappers/*.cpp)
    FOREACH(wrapperCppPath ${wrappers})
        # Generate output file name
        get_filename_component(wrapper_obj ${wrapperCppPath} NAME_WE)
        MESSAGE(STATUS "Process file: ${wrapperCppPath} -> ${wrapper_obj}")
        set_source_files_properties(${wrapperCppPath} PROPERTIES COMPILE_FLAGS "-shared -g -std=c++11 -fPIC -fno-rtti")
        add_library(${wrapper_obj} ${wrapperCppPath} )
        target_include_directories(${wrapper_obj} PUBLIC ${HALIDE_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/halide)
        target_link_libraries(${wrapper_obj} pthread dl pthread z glog gflags boost_system-mt caffe Halide)
        caffe_default_properties(${wrapper_obj})
        caffe_set_runtime_directory(${wrapper_obj} "${PROJECT_BINARY_DIR}/halide")
        install(TARGETS ${wrapper_obj} DESTINATION lib/halide)
        #caffe_set_runtime_directory(${wrapper_obj} "${PROJECT_BINARY_DIR}/test")
        #set_target_properties(${wrapper_obj} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test")

    ENDFOREACH(wrapperCppPath)


    # ---[ Adding generator target
    file(GLOB gen_srcs generator/*.cpp)
    if(gen_srcs)
        set_source_files_properties(${gen_srcs} PROPERTIES COMPILE_FLAGS "-std=c++11 -g -fno-rtti")
        link_directories(${HALIDE_BIN_DIR})
        add_executable(halide_object_generator ${gen_srcs})
       
        target_include_directories(halide_object_generator PUBLIC ${HALIDE_INCLUDE_DIR})
        target_link_libraries(halide_object_generator Halide pthread dl  pthread z)
        caffe_default_properties(halide_object_generator)
        caffe_set_runtime_directory(halide_object_generator "${PROJECT_BINARY_DIR}/halide")

        install(TARGETS halide_object_generator DESTINATION bin)
        

        file(GLOB gens generator/*.gen)
        FOREACH(infilePath ${gens})
            # Generate output file name
            get_filename_component(in_obj ${infilePath} NAME_WE)
            MESSAGE(STATUS "Process file: ${gens} -> ${in_obj}")
            SET(outfile "${PROJECT_BINARY_DIR}/halide/${in_obj}")
            MESSAGE(WARNING "Output file: ${outfile}.o")
            add_custom_target(halide_generator_run_${in_obj}
              #OUTPUT "${outfile}"
              COMMAND halide_object_generator -g ${in_obj} -o . target=cuda
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/halide
              COMMENT "run generated halide_object_generator in ${PROJECT_BINARY_DIR}/halide"
              SOURCES ${halide_object_SOURCES}
            )
            FOREACH(wrapperCppPath ${wrappers})
                # Generate output file name
                get_filename_component(wrapper_obj ${wrapperCppPath} NAME_WE)
                add_dependencies(${wrapper_obj} halide_generator_run_${in_obj})
            ENDFOREACH(wrapperCppPath)
        ENDFOREACH(infilePath)
    endif()
endif()
