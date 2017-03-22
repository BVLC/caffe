# This file differs from ../CMakeLists.txt only in the target_name_mod
# var and lack of add_subdirectory call.

if(BUILD_halide)

    #Target name modification
    #The general cmake script takes its target names from files. This variable
    #prevents having two similarly named files in one build.
    set(target_name_mod "test")

    # compile the wrapper, specify this first because we want to have their target handles
    # so that we can add all objects to it ( which we will specify next
    # this is done so that we can have a many objects to one wrapper mapping

    file(GLOB wrappers wrappers/*.cpp)
    FOREACH(wrapperCppPath ${wrappers})
        # Generate output file name
        get_filename_component(wrapper_obj ${wrapperCppPath} NAME_WE)
        MESSAGE(STATUS "Process file: ${wrapperCppPath} -> ${wrapper_obj}")
        set_source_files_properties(${wrapperCppPath} PROPERTIES COMPILE_FLAGS "-shared -g -std=c++11 -fPIC -fno-rtti")
        set( wrapper_target "${wrapper_obj}${target_name_mod}")
        add_library(${wrapper_target} ${wrapperCppPath} )
        target_include_directories(${wrapper_target} PUBLIC ${HALIDE_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/halide)
        target_link_libraries(${wrapper_target} pthread dl pthread z glog gflags boost_system-mt caffe
                              ${Halide_LINKER_LIBS})

        caffe_default_properties(${wrapper_target})
        caffe_set_runtime_directory(${wrapper_target} "${PROJECT_BINARY_DIR}/halide")
        install(TARGETS ${wrapper_target} DESTINATION lib/halide)
        #caffe_set_runtime_directory(${wrapper_target} "${PROJECT_BINARY_DIR}/test")
        #set_target_properties(${wrapper_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test")

    ENDFOREACH(wrapperCppPath)


    # ---[ Adding generator target
    file(GLOB gen_srcs generator/*.cpp)
    if(gen_srcs)
        set_source_files_properties(${gen_srcs} PROPERTIES COMPILE_FLAGS "-std=c++11 -g -fno-rtti")
        link_directories(${HALIDE_BIN_DIR})

        set(local_generator_tgt "halide_object_generator${target_name_mod}")
        add_executable(${local_generator_tgt} ${gen_srcs})

        target_include_directories(${local_generator_tgt} PUBLIC ${HALIDE_INCLUDE_DIR})
        target_link_libraries(${local_generator_tgt} pthread dl  pthread z
                              ${Halide_LINKER_LIBS})

        caffe_default_properties(${local_generator_tgt})
        caffe_set_runtime_directory(${local_generator_tgt} "${PROJECT_BINARY_DIR}/halide")

        install(TARGETS ${local_generator_tgt} DESTINATION bin)


        file(GLOB gens generator/*.gen)
        FOREACH(infilePath ${gens})
            # Generate output file name
            get_filename_component(in_obj ${infilePath} NAME_WE)
            MESSAGE(STATUS "Process file: ${gens} -> ${in_obj}")
            SET(outfile "${PROJECT_BINARY_DIR}/halide/${in_obj}")
            MESSAGE(WARNING "Output file: ${outfile}.o")
            set( in_tgt "${in_obj}${target_name_mod}")
            add_custom_target(halide_generator_run_${in_tgt}
              #OUTPUT "${outfile}"
              COMMAND ${local_generator_tgt} -g ${in_obj} -o . target=cuda
              WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/halide
              COMMENT "run generated ${local_generator_tgt} in ${PROJECT_BINARY_DIR}/halide"
              SOURCES ${halide_object_SOURCES}
            )
            FOREACH(wrapperCppPath ${wrappers})
                # Generate output file name
                get_filename_component(wrapper_traget ${wrapperCppPath} NAME_WE)
                add_dependencies(${wrapper_target} halide_generator_run_${in_tgt})
            ENDFOREACH(wrapperCppPath)
        ENDFOREACH(infilePath)
    endif()
endif()
