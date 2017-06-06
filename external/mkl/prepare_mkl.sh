#!/bin/sh
# set -ex
# 
# All modification made by Intel Corporation: © 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
FindLibrary()
{
# Find all the instances of the MKL libraries present in Caffe
  MKL_LIBS=`find $1 -name libmklml_intel.so`

# Sort libraries based on build date in $MKL/include/mkl_version.h. 
# Cut out everything but build date tagged with __INTEL_MKL_BUILD_DATE.
# The format of sorted lines is: "build_date:path_to_mkl_lib/include/mkl_version.h".
# Sort lines based on the first column (build_date), in reversed order (the recent on the top).
# Cut out include/mkl_version.h.
  RECENT_VERSION=`echo "$MKL_LIBS" \
                 | sed -e 's/lib.*$/include\/mkl_version.h/' \
                 | xargs grep __INTEL_MKL_BUILD_DATE /dev/null \
                 | sed -e 's/\(.*\):.* \([0-9]*\)/\2:\1/' \
                 | sort -rk1 -t: | head -n1 | cut -d ":" -f 2- \
                 | sed -e 's/include\/mkl_version.h//'`

# Find once again libmklml_intel.so to obtain the path with most recent MKL lib.
# TODO: obtain path from MKL_LIBS.
  RECENT_MKL=`find $RECENT_VERSION -name libmklml_intel.so`
  LOCALMKL=$RECENT_MKL
}

GetVersionName()
{
VERSION_LINE=0
if [ $1 ]; then
  VERSION_LINE=`grep __INTEL_MKL_BUILD_DATE $1/include/mkl_version.h 2>/dev/null | sed -e 's/.* //'`
fi
if [ -z $VERSION_LINE ]; then
  VERSION_LINE=0
fi
echo $VERSION_LINE  # Return Version Line
}

# MKL
DST=`dirname $0`
OMP=0 
VERSION_MATCH=20170425
ARCHIVE_BASENAME=mklml_lnx_2018.0.20170425.tgz
MKL_CONTENT_DIR=`echo $ARCHIVE_BASENAME | rev | cut -d "." -f 2- | rev`
GITHUB_RELEASE_TAG=1.0.0

MKLURL="https://github.com/intel/caffe/releases/download/$GITHUB_RELEASE_TAG/$ARCHIVE_BASENAME"
# there are diffrent MKL lib to be used for GCC and for ICC
reg='^[0-9]+$'
VERSION_LINE=`GetVersionName $MKLROOT`
# Check if MKLROOT is set if positive then set one will be used..
if [ -z $MKLROOT ] || [ $VERSION_LINE -lt $VERSION_MATCH ]; then
	# ..if MKLROOT is not set then check if we have MKL downloaded in proper version
    VERSION_LINE=`GetVersionName $DST/$MKL_CONTENT_DIR`
    if [ $VERSION_LINE -lt $VERSION_MATCH ] ; then
      #...If it is not then downloaded and unpacked
      wget --no-check-certificate -P $DST $MKLURL -O $DST/$ARCHIVE_BASENAME
      tar -xzf $DST/$ARCHIVE_BASENAME -C $DST
    fi
  FindLibrary $DST
  MKLROOT=$PWD/`echo $LOCALMKL | sed -e 's/lib.*$//'`
fi
# Check what MKL lib we have in MKLROOT
if [ -z `find $MKLROOT -name libmkl_rt.so -print -quit` ]; then
# mkl_rt has not been found; we are dealing with MKLML

  if [ -z $LOCALMKL ] ; then
# LOCALMKL is not set, when MKLROOT was set manually and it points to MKLML in correct version
    FindLibrary $MKLROOT
  fi
    
  LIBRARIES=`basename $LOCALMKL | sed -e 's/^.*lib//' | sed -e 's/\.so.*$//'`
  OMP=1
else
  LIBRARIES="mkl_rt"
fi 

# return value to calling script (Makefile,cmake)
echo $MKLROOT $LIBRARIES $OMP
