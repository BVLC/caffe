#  Adapted from cmake-modules Google Code project
#
#  Copyright (c) 2006 Andreas Schneider <mail@cynapses.org>
#
#  (Changes for libfreenect) Copyright (c) 2011 Yannis Gravezas <wizgrav@infrael.com>
#
# Redistribution and use is allowed according to the terms of the New BSD license.
#
# CMake-Modules Project New BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the CMake-Modules Project nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


if (LIBJPEG_LIBRARIES AND LIBJPEG_INCLUDE_DIRS)
  # in cache already
  set(LIBJPEG_FOUND TRUE)
else (LIBJPEG_LIBRARIES AND LIBJPEG_INCLUDE_DIRS)
  find_path(LIBJPEG_INCLUDE_DIR
    NAMES
	jpeglib.h
    PATHS
	  /opt/libjpeg-turbo/include
      /usr/include/libjpeg-turbo
      /usr/include/libjpeg
      /usr/local/include/libjpeg-turbo
      /usr/local/include/libjpeg
    PATH_SUFFIXES
	  libjpeg-turbo
	  libjpeg
  )

  find_library(LIBJPEG_LIBRARY
    NAMES
      jpeg
    PATHS
	  /opt/libjpeg-turbo/lib
      /usr/local/lib64
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
  )
  set(LIBJPEG_INCLUDE_DIRS
    ${LIBJPEG_INCLUDE_DIR}
  )
  set(LIBJPEG_LIBRARIES
    ${LIBJPEG_LIBRARY}
)
  if (LIBJPEG_INCLUDE_DIRS AND LIBJPEG_LIBRARIES)
     set(LIBJPEG_FOUND TRUE)
  endif (LIBJPEG_INCLUDE_DIRS AND LIBJPEG_LIBRARIES)

  if (LIBJPEG_FOUND)
    if (NOT libjpeg_FIND_QUIETLY)
      message(STATUS "Found libjpeg:")
	  message(STATUS " - Includes: ${LIBJPEG_INCLUDE_DIRS}")
	  message(STATUS " - Libraries: ${LIBJPEG_LIBRARIES}")
	  
    endif (NOT libjpeg_FIND_QUIETLY)
  else (LIBJPEG_FOUND)
    if (libjpeg_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find libjpeg")
    endif (libjpeg_FIND_REQUIRED)
  endif (LIBJPEG_FOUND)

  # show the LIBFREENECT_INCLUDE_DIRS and LIBFREENECT_LIBRARIES variables only in the advanced view
  mark_as_advanced(LIBJPEG_INCLUDE_DIRS LIBJPEG_LIBRARIES)

endif (LIBJPEG_LIBRARIES AND LIBJPEG_INCLUDE_DIRS)
