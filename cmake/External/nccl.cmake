# if (NOT __NCCL_INCLUDED) # guard against multiple includes
  set(__NCCL_INCLUDED TRUE)
  if(MSVC)
    # use the system-wide nccl if present
    find_package(NCCL)
    if (NCCL_FOUND)
        set(NCCL_EXTERNAL FALSE)
    else()
        # build directory
        set(nccl_PREFIX ${CMAKE_BINARY_DIR}/external/nccl-prefix)
        # install directory
        set(nccl_INSTALL ${CMAKE_BINARY_DIR}/external/nccl-install)
        ExternalProject_Add(nccl
        PREFIX ${nccl_PREFIX}
        URL https://github.com/willyd/nccl/archive/470b3130457f125f4608c7baee71123aa16a3b12.zip
        UPDATE_COMMAND ""
        INSTALL_DIR ${nccl_INSTALL}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_INSTALL_PREFIX=${nccl_INSTALL}
                   -DBUILD_SHARED_LIBS=OFF
                   -DNCCL_BUILD_TESTS:BOOL=OFF

        LOG_DOWNLOAD 1
        LOG_INSTALL 1
        BUILD_BYPRODUCTS ${nccl_INSTALL}/include ${nccl_INSTALL}/lib/nccl.lib
        )

        set(NCCL_INCLUDE_DIR ${nccl_INSTALL}/include)
        set(NCCL_LIBRARIES ${nccl_INSTALL}/lib/nccl.lib)
    endif()
  else()
    # default to find package on UNIX systems
    find_package(NCCL REQUIRED)
  endif()
# endif()