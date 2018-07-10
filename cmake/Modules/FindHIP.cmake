###############################################################################
# FindHIP.cmake
###############################################################################

###############################################################################
# SET: Variable defaults
###############################################################################
# User defined flags
set(HIP_HIPCC_FLAGS "" CACHE STRING "Semicolon delimited flags for HIPCC")
set(HIP_HCC_FLAGS "" CACHE STRING "Semicolon delimited flags for HCC")
set(HIP_NVCC_FLAGS "" CACHE STRING "Semicolon delimted flags for NVCC")
mark_as_advanced(HIP_HIPCC_FLAGS HIP_HCC_FLAGS HIP_NVCC_FLAGS)
set(_hip_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
list(REMOVE_DUPLICATES _hip_configuration_types)
foreach(config ${_hip_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(HIP_HIPCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for HIPCC")
    set(HIP_HCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for HCC")
    set(HIP_NVCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for NVCC")
    mark_as_advanced(HIP_HIPCC_FLAGS_${config_upper} HIP_HCC_FLAGS_${config_upper} HIP_NVCC_FLAGS_${config_upper})
endforeach()
option(HIP_HOST_COMPILATION_CPP "Host code compilation mode" ON)
mark_as_advanced(HIP_HOST_COMPILATION_CPP)

###############################################################################
# FIND: HIP and associated helper binaries
###############################################################################
# HIP is supported on Linux only
if(UNIX AND NOT APPLE AND NOT CYGWIN)
    # Search for HIP installation
    if(NOT HIP_ROOT_DIR)
        # Search in user specified path first
        find_path(
            HIP_ROOT_DIR
            NAMES hipconfig
            PATHS
            ENV ROCM_PATH
            ENV HIP_PATH
            PATH_SUFFIXES bin
            DOC "HIP installed location"
            NO_DEFAULT_PATH
            )
        # Now search in default path
        find_path(
            HIP_ROOT_DIR
            NAMES hipconfig
            PATHS
            /opt/rocm
            /opt/rocm/hip
            PATH_SUFFIXES bin
            DOC "HIP installed location"
            )

        # Check if we found HIP installation
        if(HIP_ROOT_DIR)
            # If so, fix the path
            string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" HIP_ROOT_DIR ${HIP_ROOT_DIR})
            # And push it back to the cache
            set(HIP_ROOT_DIR ${HIP_ROOT_DIR} CACHE PATH "HIP installed location" FORCE)
        endif()
        if(NOT EXISTS ${HIP_ROOT_DIR})
            if(HIP_FIND_REQUIRED)
                message(FATAL_ERROR "Specify HIP_ROOT_DIR")
            elseif(NOT HIP_FIND_QUIETLY)
                message("HIP_ROOT_DIR not found or specified")
            endif()
        endif()
    endif()

    # Find HIPCC executable
    find_program(
        HIP_HIPCC_EXECUTABLE
        NAMES hipcc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    if(NOT HIP_HIPCC_EXECUTABLE)
        # Now search in default paths
        find_program(HIP_HIPCC_EXECUTABLE hipcc)
    endif()
    mark_as_advanced(HIP_HIPCC_EXECUTABLE)

    # Find HIPCONFIG executable
    find_program(
        HIP_HIPCONFIG_EXECUTABLE
        NAMES hipconfig
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    if(NOT HIP_HIPCONFIG_EXECUTABLE)
        # Now search in default paths
        find_program(HIP_HIPCONFIG_EXECUTABLE hipconfig)
    endif()
    mark_as_advanced(HIP_HIPCONFIG_EXECUTABLE)

    # Find HIPCC_CMAKE_LINKER_HELPER executable
    find_program(
        HIP_HIPCC_CMAKE_LINKER_HELPER
        NAMES hipcc_cmake_linker_helper
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    if(NOT HIP_HIPCC_CMAKE_LINKER_HELPER)
        # Now search in default paths
        find_program(HIP_HIPCC_CMAKE_LINKER_HELPER hipcc_cmake_linker_helper)
    endif()
    mark_as_advanced(HIP_HIPCC_CMAKE_LINKER_HELPER)

    if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_VERSION)
        # Compute the version
        execute_process(
            COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
            OUTPUT_VARIABLE _hip_version
            ERROR_VARIABLE _hip_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
            )
        if(NOT _hip_error)
            set(HIP_VERSION ${_hip_version} CACHE STRING "Version of HIP as computed from hipcc")
        else()
            set(HIP_VERSION "0.0.0" CACHE STRING "Version of HIP as computed by FindHIP()")
        endif()
        mark_as_advanced(HIP_VERSION)
    endif()
    if(HIP_VERSION)
        string(REPLACE "." ";" _hip_version_list "${HIP_VERSION}")
        list(GET _hip_version_list 0 HIP_VERSION_MAJOR)
        list(GET _hip_version_list 1 HIP_VERSION_MINOR)
        list(GET _hip_version_list 2 HIP_VERSION_PATCH)
        set(HIP_VERSION_STRING "${HIP_VERSION}")
    endif()

    if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_PLATFORM)
        # Compute the platform
        execute_process(
            COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --platform
            OUTPUT_VARIABLE _hip_platform
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        set(HIP_PLATFORM ${_hip_platform} CACHE STRING "HIP platform as computed by hipconfig")
        mark_as_advanced(HIP_PLATFORM)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    HIP
    REQUIRED_VARS
    HIP_ROOT_DIR
    HIP_HIPCC_EXECUTABLE
    HIP_HIPCONFIG_EXECUTABLE
    HIP_PLATFORM
    VERSION_VAR HIP_VERSION
    )

###############################################################################
# MACRO: Locate helper files
###############################################################################
macro(HIP_FIND_HELPER_FILE _name _extension)
    set(_hip_full_name "${_name}.${_extension}")
    get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
    set(HIP_${_name} "${CMAKE_CURRENT_LIST_DIR}/FindHIP/${_hip_full_name}")
    if(NOT EXISTS "${HIP_${_name}}")
        set(error_message "${_hip_full_name} not found in ${CMAKE_CURRENT_LIST_DIR}/FindHIP")
        if(HIP_FIND_REQUIRED)
            message(FATAL_ERROR "${error_message}")
        else()
            if(NOT HIP_FIND_QUIETLY)
                message(STATUS "${error_message}")
            endif()
        endif()
    endif()
    # Set this variable as internal, so the user isn't bugged with it.
    set(HIP_${_name} ${HIP_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro()

###############################################################################
hip_find_helper_file(run_make2cmake cmake)
hip_find_helper_file(run_hipcc cmake)
###############################################################################

###############################################################################
# MACRO: Reset compiler flags
###############################################################################
macro(HIP_RESET_FLAGS)
    unset(HIP_HIPCC_FLAGS)
    unset(HIP_HCC_FLAGS)
    unset(HIP_NVCC_FLAGS)
    foreach(config ${_hip_configuration_types})
        string(TOUPPER ${config} config_upper)
        unset(HIP_HIPCC_FLAGS_${config_upper})
        unset(HIP_HCC_FLAGS_${config_upper})
        unset(HIP_NVCC_FLAGS_${config_upper})
    endforeach()
endmacro()

###############################################################################
# MACRO: Separate the options from the sources
###############################################################################
macro(HIP_GET_SOURCES_AND_OPTIONS _sources _cmake_options _hipcc_options _hcc_options _nvcc_options)
    set(${_sources})
    set(${_cmake_options})
    set(${_hipcc_options})
    set(${_hcc_options})
    set(${_nvcc_options})
    set(_hipcc_found_options FALSE)
    set(_hcc_found_options FALSE)
    set(_nvcc_found_options FALSE)
    foreach(arg ${ARGN})
        if("x${arg}" STREQUAL "xHIPCC_OPTIONS")
            set(_hipcc_found_options TRUE)
            set(_hcc_found_options FALSE)
            set(_nvcc_found_options FALSE)
        elseif("x${arg}" STREQUAL "xHCC_OPTIONS")
            set(_hipcc_found_options FALSE)
            set(_hcc_found_options TRUE)
            set(_nvcc_found_options FALSE)
        elseif("x${arg}" STREQUAL "xNVCC_OPTIONS")
            set(_hipcc_found_options FALSE)
            set(_hcc_found_options FALSE)
            set(_nvcc_found_options TRUE)
        elseif(
                "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
                "x${arg}" STREQUAL "xSTATIC" OR
                "x${arg}" STREQUAL "xSHARED" OR
                "x${arg}" STREQUAL "xMODULE"
                )
            list(APPEND ${_cmake_options} ${arg})
        else()
            if(_hipcc_found_options)
                list(APPEND ${_hipcc_options} ${arg})
            elseif(_hcc_found_options)
                list(APPEND ${_hcc_options} ${arg})
            elseif(_nvcc_found_options)
                list(APPEND ${_nvcc_options} ${arg})
            else()
                # Assume this is a file
                list(APPEND ${_sources} ${arg})
            endif()
        endif()
    endforeach()
endmacro()

###############################################################################
# MACRO: Add include directories to pass to the hipcc command
###############################################################################
set(HIP_HIPCC_INCLUDE_ARGS_USER "")
macro(HIP_INCLUDE_DIRECTORIES)
    foreach(dir ${ARGN})
        list(APPEND HIP_HIPCC_INCLUDE_ARGS_USER -I${dir})
    endforeach()
endmacro()

###############################################################################
# FUNCTION: Helper to avoid clashes of files with the same basename but different paths
###############################################################################
function(HIP_COMPUTE_BUILD_PATH path build_path)
    # Convert to cmake style paths
    file(TO_CMAKE_PATH "${path}" bpath)
    if(IS_ABSOLUTE "${bpath}")
        string(FIND "${bpath}" "${CMAKE_CURRENT_BINARY_DIR}" _binary_dir_pos)
        if(_binary_dir_pos EQUAL 0)
            file(RELATIVE_PATH bpath "${CMAKE_CURRENT_BINARY_DIR}" "${bpath}")
        else()
            file(RELATIVE_PATH bpath "${CMAKE_CURRENT_SOURCE_DIR}" "${bpath}")
        endif()
    endif()

    # Remove leading /
    string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
    # Avoid absolute paths by removing ':'
    string(REPLACE ":" "_" bpath "${bpath}")
    # Avoid relative paths that go up the tree
    string(REPLACE "../" "__/" bpath "${bpath}")
    # Avoid spaces
    string(REPLACE " " "_" bpath "${bpath}")
    # Strip off the filename
    get_filename_component(bpath "${bpath}" PATH)

    set(${build_path} "${bpath}" PARENT_SCOPE)
endfunction()

###############################################################################
# MACRO: Parse OPTIONS from ARGN & set variables prefixed by _option_prefix
###############################################################################
macro(HIP_PARSE_HIPCC_OPTIONS _option_prefix)
    set(_hip_found_config)
    foreach(arg ${ARGN})
        # Determine if we are dealing with a per-configuration flag
        foreach(config ${_hip_configuration_types})
            string(TOUPPER ${config} config_upper)
            if(arg STREQUAL "${config_upper}")
                set(_hip_found_config _${arg})
                # Clear arg to prevent it from being processed anymore
                set(arg)
            endif()
        endforeach()
        if(arg)
            list(APPEND ${_option_prefix}${_hip_found_config} "${arg}")
        endif()
    endforeach()
endmacro()

###############################################################################
# MACRO: Try and include dependency file if it exists
###############################################################################
macro(HIP_INCLUDE_HIPCC_DEPENDENCIES dependency_file)
    set(HIP_HIPCC_DEPEND)
    set(HIP_HIPCC_DEPEND_REGENERATE FALSE)

    # Create the dependency file if it doesn't exist
    if(NOT EXISTS ${dependency_file})
        file(WRITE ${dependency_file} "# Generated by: FindHIP.cmake. Do not edit.\n")
    endif()
    # Include the dependency file
    include(${dependency_file})

    # Verify the existence of all the included files
    if(HIP_HIPCC_DEPEND)
        foreach(f ${HIP_HIPCC_DEPEND})
            if(NOT EXISTS ${f})
                # If they aren't there, regenerate the file again
                set(HIP_HIPCC_DEPEND_REGENERATE TRUE)
            endif()
        endforeach()
    else()
        # No dependencies, so regenerate the file
        set(HIP_HIPCC_DEPEND_REGENERATE TRUE)
    endif()

    # Regenerate the dependency file if needed
    if(HIP_HIPCC_DEPEND_REGENERATE)
        set(HIP_HIPCC_DEPEND ${dependency_file})
        file(WRITE ${dependency_file} "# Generated by: FindHIP.cmake. Do not edit.\n")
    endif()
endmacro()

###############################################################################
# MACRO: Prepare cmake commands for the target
###############################################################################
macro(HIP_PREPARE_TARGET_COMMANDS _target _format _generated_files _source_files)
    set(_hip_flags "")
    set(_hip_build_configuration "${CMAKE_BUILD_TYPE}")
    if(HIP_HOST_COMPILATION_CPP)
        set(HIP_C_OR_CXX CXX)
    else()
        set(HIP_C_OR_CXX C)
    endif()
    set(generated_extension ${CMAKE_${HIP_C_OR_CXX}_OUTPUT_EXTENSION})

    # Initialize list of includes with those specified by the user. Append with
    # ones specified to cmake directly.
    set(HIP_HIPCC_INCLUDE_ARGS ${HIP_HIPCC_INCLUDE_ARGS_USER})
    get_directory_property(_hip_include_directories INCLUDE_DIRECTORIES)
    list(REMOVE_DUPLICATES _hip_include_directories)
    if(_hip_include_directories)
        foreach(dir ${_hip_include_directories})
            list(APPEND HIP_HIPCC_INCLUDE_ARGS -I${dir})
        endforeach()
    endif()

    HIP_GET_SOURCES_AND_OPTIONS(_hip_sources _hip_cmake_options _hipcc_options _hcc_options _nvcc_options ${ARGN})
    HIP_PARSE_HIPCC_OPTIONS(HIP_HIPCC_FLAGS ${_hipcc_options})
    HIP_PARSE_HIPCC_OPTIONS(HIP_HCC_FLAGS ${_hcc_options})
    HIP_PARSE_HIPCC_OPTIONS(HIP_NVCC_FLAGS ${_nvcc_options})

    # Check if we are building shared library.
    set(_hip_build_shared_libs FALSE)
    list(FIND _hip_cmake_options SHARED _hip_found_SHARED)
    list(FIND _hip_cmake_options MODULE _hip_found_MODULE)
    if(_hip_found_SHARED GREATER -1 OR _hip_found_MODULE GREATER -1)
        set(_hip_build_shared_libs TRUE)
    endif()
    list(FIND _hip_cmake_options STATIC _hip_found_STATIC)
    if(_hip_found_STATIC GREATER -1)
        set(_hip_build_shared_libs FALSE)
    endif()

    # If we are building a shared library, add extra flags to HIP_HIPCC_FLAGS
    if(_hip_build_shared_libs)
        list(APPEND HIP_HCC_FLAGS "-fPIC")
        list(APPEND HIP_NVCC_FLAGS "--shared -Xcompiler '-fPIC'")
    endif()

    # Set host compiler
    set(HIP_HOST_COMPILER "${CMAKE_${HIP_C_OR_CXX}_COMPILER}")

    # Set compiler flags
    set(_HIP_HOST_FLAGS "set(CMAKE_HOST_FLAGS ${CMAKE_${HIP_C_OR_CXX}_FLAGS})")
    set(_HIP_HIPCC_FLAGS "set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS})")
    set(_HIP_HCC_FLAGS "set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS})")
    set(_HIP_NVCC_FLAGS "set(HIP_NVCC_FLAGS ${HIP_NVCC_FLAGS})")
    foreach(config ${_hip_configuration_types})
        string(TOUPPER ${config} config_upper)
        set(_HIP_HOST_FLAGS "${_HIP_HOST_FLAGS}\nset(CMAKE_HOST_FLAGS_${config_upper} ${CMAKE_${HIP_C_OR_CXX}_FLAGS_${config_upper}})")
        set(_HIP_HIPCC_FLAGS "${_HIP_HIPCC_FLAGS}\nset(HIP_HIPCC_FLAGS_${config_upper} ${HIP_HIPCC_FLAGS_${config_upper}})")
        set(_HIP_HCC_FLAGS "${_HIP_HCC_FLAGS}\nset(HIP_HCC_FLAGS_${config_upper} ${HIP_HCC_FLAGS_${config_upper}})")
        set(_HIP_NVCC_FLAGS "${_HIP_NVCC_FLAGS}\nset(HIP_NVCC_FLAGS_${config_upper} ${HIP_NVCC_FLAGS_${config_upper}})")
    endforeach()

    # Reset the output variable
    set(_hip_generated_files "")
    set(_hip_source_files "")

    # Iterate over all arguments and create custom commands for all source files
    foreach(file ${ARGN})
        # Ignore any file marked as a HEADER_FILE_ONLY
        get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
        # Allow per source file overrides of the format. Also allows compiling non .cu files.
        get_source_file_property(_hip_source_format ${file} HIP_SOURCE_PROPERTY_FORMAT)
        if((${file} MATCHES "\\.cu$" OR _hip_source_format) AND NOT _is_header)
            set(host_flag FALSE)
        else()
            set(host_flag TRUE)
        endif()

        if(NOT host_flag)
            # Determine output directory
            HIP_COMPUTE_BUILD_PATH("${file}" hip_build_path)
            set(hip_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}.dir/${hip_build_path}")

            get_filename_component(basename ${file} NAME)
            set(generated_file_path "${hip_compile_output_dir}/${CMAKE_CFG_INTDIR}")
            set(generated_file_basename "${_target}_generated_${basename}${generated_extension}")

            # Set file names
            set(generated_file "${generated_file_path}/${generated_file_basename}")
            set(cmake_dependency_file "${hip_compile_output_dir}/${generated_file_basename}.depend")
            set(custom_target_script_pregen "${hip_compile_output_dir}/${generated_file_basename}.cmake.pre-gen")
            set(custom_target_script "${hip_compile_output_dir}/${generated_file_basename}.cmake")

            # Set properties for object files
            set_source_files_properties("${generated_file}"
                PROPERTIES
                EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked
                )

            # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path
            get_filename_component(file_path "${file}" PATH)
            if(IS_ABSOLUTE "${file_path}")
                set(source_file "${file}")
            else()
                set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
            endif()

            # Bring in the dependencies
            HIP_INCLUDE_HIPCC_DEPENDENCIES(${cmake_dependency_file})

            # Configure the build script
            configure_file("${HIP_run_hipcc}" "${custom_target_script_pregen}" @ONLY)
            file(GENERATE
                OUTPUT "${custom_target_script}"
                INPUT "${custom_target_script_pregen}"
                )
            set(main_dep DEPENDS ${source_file})
            set(verbose_output "$(VERBOSE)")

            # Create up the comment string
            file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
            set(hip_build_comment_string "Building HIPCC object ${generated_file_relative_path}")

            # Build the generated file and dependency file
            add_custom_command(
                OUTPUT ${generated_file}
                # These output files depend on the source_file and the contents of cmake_dependency_file
                ${main_dep}
                DEPENDS ${HIP_HIPCC_DEPEND}
                DEPENDS ${custom_target_script}
                # Make sure the output directory exists before trying to write to it.
                COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
                COMMAND ${CMAKE_COMMAND} ARGS
                -D verbose:BOOL=${verbose_output}
                -D build_configuration:STRING=${_hip_build_configuration}
                -D "generated_file:STRING=${generated_file}"
                -P "${custom_target_script}"
                WORKING_DIRECTORY "${hip_compile_output_dir}"
                COMMENT "${hip_build_comment_string}"
                )

            # Make sure the build system knows the file is generated
            set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)
            list(APPEND _hip_generated_files ${generated_file})
            list(APPEND _hip_source_files ${file})
        endif()
    endforeach()

    # Set the return parameter
    set(${_generated_files} ${_hip_generated_files})
    set(${_source_files} ${_hip_source_files})
endmacro()

###############################################################################
# HIP_ADD_EXECUTABLE
###############################################################################
macro(HIP_ADD_EXECUTABLE hip_target)
    # Separate the sources from the options
    HIP_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _hipcc_options _hcc_options _nvcc_options ${ARGN})
    HIP_PREPARE_TARGET_COMMANDS(${hip_target} OBJ _generated_files _source_files ${_sources} HIPCC_OPTIONS ${_hipcc_options} HCC_OPTIONS ${_hcc_options} NVCC_OPTIONS ${_nvcc_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()
    if("x${HCC_HOME}" STREQUAL "x")
        set(HCC_HOME "/opt/rocm/hcc")
    endif()
    set(CMAKE_HIP_LINK_EXECUTABLE "${HIP_HIPCC_CMAKE_LINKER_HELPER} ${HCC_HOME} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    add_executable(${hip_target} ${_cmake_options} ${_generated_files} ${_sources})
    set_target_properties(${hip_target} PROPERTIES LINKER_LANGUAGE HIP)
endmacro()

###############################################################################
# HIP_ADD_LIBRARY
###############################################################################
macro(HIP_ADD_LIBRARY hip_target)
    # Separate the sources from the options
    HIP_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _hipcc_options _hcc_options _nvcc_options ${ARGN})
    HIP_PREPARE_TARGET_COMMANDS(${hip_target} OBJ _generated_files _source_files ${_sources} ${_cmake_options} HIPCC_OPTIONS ${_hipcc_options} HCC_OPTIONS ${_hcc_options} NVCC_OPTIONS ${_nvcc_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()
    add_library(${hip_target} ${_cmake_options} ${_generated_files} ${_sources})
    set_target_properties(${hip_target} PROPERTIES LINKER_LANGUAGE ${HIP_C_OR_CXX})
endmacro()

# vim: ts=4:sw=4:expandtab:smartindent