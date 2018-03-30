function(Download_DLCP)
  find_program(HAS_ICPC NAMES icpc DOC "Intel Compiler")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(DLCP_CXX "${CMAKE_CXX_COMPILER}" PARENT_SCOPE)
  elseif(HAS_ICPC)
    set(DLCP_CXX "icpc" PARENT_SCOPE)
  else()
    message("weight grad compression is disabled because intel compiler is not found")
    return()  
  endif()  

  set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
  set(DLCP_IDEEPDIR ${EXTERNAL_DIR}/ideep)
  set(DLCP_ROOTDIR ${DLCP_IDEEPDIR}/dlcp)
  set(DLCP_INCLDIR "${DLCP_ROOTDIR}/include" PARENT_SCOPE)
  set(DLCP_LIBDIR ${DLCP_ROOTDIR}/lib PARENT_SCOPE)
 
  # Download dl compression lib source code if it doesn't exist 
  if (NOT EXISTS ${DLCP_INCLDIR}/dl_compression.h)
    execute_process(COMMAND rm -rf ${DLCP_IDEEPDIR})
    execute_process(COMMAND git clone https://github.com/intel/ideep.git -b ideep4py ${DLCP_IDEEPDIR})
  endif()

  add_custom_target(DLCP_Build ALL
                    COMMAND export DLCP_CXX=${DLCP_CXX}
                    COMMAND make -j
                    WORKING_DIRECTORY ${DLCP_ROOTDIR})

endfunction(Download_DLCP)
