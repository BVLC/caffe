# This file differs from ../CMakeLists.txt only in the target_name_mod
# var and lack of add_subdirectory call.

if(BUILD_halide)
    # Target name modification:
    # In general take target names from *.gen files. This variable would allow
    # chaning names dependent on which subdir a copy of this cmake file is in.
    # This could be used to speparate test functions with the same name as
    # non test-functions.
    set(target_name_mod "")

    # ---[ Create generator executable
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
    endif()


    # ---[ Run generator executable to generate halide objects
    set(Halide_OBJECTS "")
    file(GLOB gens generator/*.gen)
    foreach(infilePath ${gens})
        # Generate output file name
        get_filename_component(in_obj ${infilePath} NAME_WE)
        message(STATUS "Halide build: found generator ${in_obj} at ${infilePath}")
        SET(outfile "${PROJECT_BINARY_DIR}/halide/${in_obj}.o")
        set( in_tgt "${in_obj}${target_name_mod}")

        # add_custom_target is of type UTILITY which we cant linked
        add_custom_target(
          halide_generator_run_${in_tgt}
          #OUTPUT ${outfile}
          COMMAND ${local_generator_tgt} -g ${in_obj} target=cuda -e o,h -o .
          WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/halide
          COMMENT "ran generated ${local_generator_tgt} in ${PROJECT_BINARY_DIR}/halide"
          SOURCES ${halide_object_SOURCES}
        )
        # so import the outputs and treat them like a library
        add_library( ${in_tgt} STATIC IMPORTED GLOBAL)
        set_target_properties(${in_tgt}
            PROPERTIES
            IMPORTED_LOCATION ${outfile}
            INTERFACE_INCLUDE_DIRECTORIES  ${PROJECT_BINARY_DIR}/halide/)
        # make sure we run the generator
        add_dependencies(${in_tgt} halide_generator_run_${in_tgt})
        list(APPEND Halide_OBJECTS ${in_tgt})
    endforeach(infilePath)

    # ---[ Compile halide wrapers and link in halide objects
    file(GLOB wrappers wrappers/*.cpp)
    foreach(wrapperCppPath ${wrappers})
        # Generate output file name
        get_filename_component(wrapper_obj ${wrapperCppPath} NAME_WE)
        message(STATUS "Halide build: found wrapper ${wrapper_obj} at ${wrapperCppPath}")
        set_source_files_properties(${wrapperCppPath} PROPERTIES COMPILE_FLAGS "-shared -g -std=c++11 -fPIC -fno-rtti")
        set( wrapper_target "${wrapper_obj}${target_name_mod}")
        add_library(${wrapper_target} ${wrapperCppPath} )
        target_include_directories(${wrapper_target} PUBLIC ${HALIDE_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/halide)
        # Add extra wrapper depedancies here.
        target_link_libraries(${wrapper_target} caffe ${Halide_LINKER_LIBS})
        caffe_default_properties(${wrapper_target})
        caffe_set_runtime_directory(${wrapper_target} "${PROJECT_BINARY_DIR}/halide")
        install(TARGETS ${wrapper_target} DESTINATION lib/halide)
        add_dependencies(${wrapper_target} ${Halide_OBJECTS})
        target_link_libraries(${wrapper_target} ${Halide_OBJECTS})
    endforeach(wrapperCppPath)
    message(STATUS "Halide build info: wrappers linked with: ${Halide_OBJECTS}")
endif()
