#!/bin/sh
# set -ex
FindLibrary() 
{
  case "$1" in
    intel|1)
      LOCALMKL=`find $DST -name libmklml_intel.so`   # name of MKL SDL lib
      ;;
    *)
      LOCALMKL=`find $DST -name libmklml_gnu.so`   # name of MKL SDL lib
      ;;
  esac

}
# MKL
DST=`dirname $0`
OMP=0 
MKLURL="https://github.com/intelcaffe/caffe/releases/download/self_contained_BU1/mklml_lnx_2017.0.b1.20160513.1.tgz"
if [ $MKLROOT ]; then
  VERSION_LINE=`grep __INTEL_MKL_BUILD_DATE $MKLROOT/include/mkl_version.h 2>/dev/null | sed -e 's/.* //'`
fi
# there are diffrent MKL lib to be used for GCC and for ICC
FindLibrary $1
reg='^[0-9]+$'
if [ -z $VERSION_LINE ]; then
  VERSION_LINE=0
else
  if ! [[ $VERSION_LINE =~ $reg ]]; then
    VERSION_LINE=0
  fi
fi
# Check if MKL_ROOT is set if positive then set one will be used..
if [ -z $MKLROOT ] || [ $VERSION_LINE -lt 20160514 ]; then
	# ..if MKLROOT is not set then check if we have MKL downloaded..
    if [ -z $LOCALMKL ] || [ ! -f $LOCALMKL ]; then
      #...If it is not then downloaded and unpacked
      wget --no-check-certificate -P $DST $MKLURL
      tar -xzf $DST/mklml_lnx*.tgz -C $DST
	  FindLibrary $1
    fi
  # set MKL env vars are to be done via generated script
  # this will help us export MKL env to existing shell
  
  MKLROOT=$PWD/`echo $LOCALMKL | sed -e 's/lib.*$//'`
fi

# Check what MKL lib we have in MKLROOT
if [ -z `find $MKLROOT -name libmkl_rt.so -print -quit` ]; then
  LIBRARIES=`basename $LOCALMKL | sed -e 's/^.*lib//' | sed -e 's/\.so.*$//'`
  OMP=1
else
  LIBRARIES="mkl_rt"
fi 


# return value to calling script (Makefile,cmake)
echo $MKLROOT $LIBRARIES $OMP
