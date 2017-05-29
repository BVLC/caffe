set(_windows_create_link_header "${CMAKE_CURRENT_LIST_FILE}")

# function to add a post build command to create a link header
function(windows_create_link_header target outputfile)
    add_custom_command(TARGET ${target} POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                                #-DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                                -DMSVC_VERSION=${MSVC_VERSION}
                                -DTARGET_FILE=$<TARGET_FILE:${target}>
                                #-DPROJECT_BINARY_DIR=${PROJECT_BINARY_DIR}
                                #-DCMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
                                #-DCONFIGURATION=$<CONFIGURATION>
                                -DOUTPUT_FILE=${outputfile}
                                -P ${_windows_create_link_header}
                        BYPRODUCTS ${outputfile}
                      )
endfunction()


function(find_dumpbin var)
    # MSVC_VERSION =
    # 1200 = VS  6.0
    # 1300 = VS  7.0
    # 1310 = VS  7.1
    # 1400 = VS  8.0
    # 1500 = VS  9.0
    # 1600 = VS 10.0
    # 1700 = VS 11.0
    # 1800 = VS 12.0
    # 1900 = VS 14.0
    set(MSVC_PRODUCT_VERSION_1200 6.0)
    set(MSVC_PRODUCT_VERSION_1300 7.0)
    set(MSVC_PRODUCT_VERSION_1310 7.1)
    set(MSVC_PRODUCT_VERSION_1400 8.0)
    set(MSVC_PRODUCT_VERSION_1500 9.0)
    set(MSVC_PRODUCT_VERSION_1600 10.0)
    set(MSVC_PRODUCT_VERSION_1700 11.0)
    set(MSVC_PRODUCT_VERSION_1800 12.0)
    set(MSVC_PRODUCT_VERSION_1900 14.0)
    get_filename_component(MSVC_VC_DIR [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\${MSVC_PRODUCT_VERSION_${MSVC_VERSION}}\\Setup\\VC;ProductDir] REALPATH CACHE)

    find_program(DUMPBIN_EXECUTABLE dumpbin ${MSVC_VC_DIR}/bin)
    if(NOT DUMPBIN_EXECUTABLE)
        message(FATAL_ERROR "Could not find DUMPBIN_EXECUTABLE please define this variable")
    endif()
    set(${var} ${DUMPBIN_EXECUTABLE} PARENT_SCOPE)
endfunction()

macro(print_date)
    execute_process(COMMAND powershell -NoProfile -Command "get-date")
endmacro()


if(CMAKE_SCRIPT_MODE_FILE)
    cmake_policy(SET CMP0007 NEW)
    # find the dumpbin exe
    find_dumpbin(dumpbin)
    # execute dumpbin to generate a list of symbols
    execute_process(COMMAND ${dumpbin} /SYMBOLS ${TARGET_FILE}
                    RESULT_VARIABLE _result
                    OUTPUT_VARIABLE _output
                    ERROR_VARIABLE _error
    )
    # match all layers and solvers instantiation guard
    string(REGEX MATCHALL "\\?gInstantiationGuard[^\\(\\) ]*" __symbols ${_output})
    # define a string to generate a list of pragmas
    foreach(__symbol ${__symbols})
        set(__pragma "${__pragma}#pragma comment(linker, \"/include:${__symbol}\")\n")        
    endforeach()
    file(WRITE ${OUTPUT_FILE} ${__pragma})
endif()

