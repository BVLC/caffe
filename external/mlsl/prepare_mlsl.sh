#!/bin/sh
# set -ex
# 
# All modification made by Intel Corporation: © 2018 Intel Corporation
# 
# This script is used to prepare the Intel® Machine Learning Scaling Library
#
FindLibrary()
{
# Find all the instances of the MLSL libraries present in Caffe
  MLSL_LIBS=`find $1 -name libmlsl.so`
  #echo "[Debug][FindLibrary function] MLSL_LIBS: $MLSL_LIBS"

  LOCALMLSL=$MLSL_LIBS
  #echo "[Debug][FindLibrary function] LOCALMLSL: $LOCALMLSL"
}

GetVersionName()
{
VERSION_LINE=0
if [ $1 ]; then
  RAW_VERSION_LINE=`echo $1 | rev | cut -d "_" -f -1 | rev`
  VERSION_LINE=`echo $RAW_VERSION_LINE | sed 's/\.//g'`
fi
if [ -z $VERSION_LINE ]; then
  VERSION_LINE=0
fi
if [ -z "$(echo $VERSION_LINE | sed -n "/^[0-9]\+$/p")" ]; then 
  #echo "[Debug] VERSION_LINE value contains other string or flags, not only numbers"
  VERSION_LINE=0  
fi 
echo $VERSION_LINE  # Return Version Line
}

# Clean up the previous MLSL version
CleanUpPreviousMLSL2017_0_014()
{
OLD_ARCHIVE_TARGZ=files.tar.gz
OLD_INSTALL_SHELL=install.sh
OLD_ARCHIVE_BASENAME=l_mlsl_p_2017.0.014.tgz
OLD_ARCHIVE_INSTALL_FOLDERNAME=l_mlsl_p_2017.0.014
if [ -f $ABS_DST/$OLD_ARCHIVE_TARGZ ]; then
  rm $ABS_DST/$OLD_ARCHIVE_TARGZ
  #echo "[Debug] Delete old files.tar.gz!"
fi
if [ -f $ABS_DST/$OLD_INSTALL_SHELL ]; then
  rm $ABS_DST/$OLD_INSTALL_SHELL
  #echo "[Debug] Delete old install.sh file!"
fi
if [ -f $ABS_DST/$OLD_ARCHIVE_BASENAME ]; then
  rm $ABS_DST/$OLD_ARCHIVE_BASENAME
  #echo "[Debug] Delete old l_mlsl_p_2017.0.014.tgz file!"
fi
if [ -d $ABS_DST/$OLD_ARCHIVE_INSTALL_FOLDERNAME ]; then 
  rm -rf $ABS_DST/$OLD_ARCHIVE_INSTALL_FOLDERNAME
  #echo "[Debug] Delete old l_mlsl_p_2017.0.014 folder!"
fi
}

# Clean up the previous MLSL version
# Can be used for l_mlsl_2017.1.016, l_mlsl_2017.2.018, l_mlsl_2018.0.003
CleanUpPreviousMLSL()
{
version_year=$1
version_num=$2
subversion_num=$3
OLD_ARCHIVE_TARGZ=files.tar.gz
OLD_ARCHIVE_BASENAME=l_mlsl_$version_year.$version_num.$subversion_num.tgz
OLD_ARCHIVE_INSTALL_FOLDERNAME=l_mlsl_$version_year.$version_num.$subversion_num
if [ -f $ABS_DST/$OLD_ARCHIVE_TARGZ ]; then
  rm $ABS_DST/$OLD_ARCHIVE_TARGZ
  #echo "[Debug] Delete old files.tar.gz!"
fi
if [ -f $ABS_DST/$OLD_ARCHIVE_BASENAME ]; then
  rm $ABS_DST/$OLD_ARCHIVE_BASENAME
  #echo "[Debug] Delete old l_mlsl_$version_year.$version_num.$subversion_num.tgz file!"
fi
if [ -d $ABS_DST/$OLD_ARCHIVE_INSTALL_FOLDERNAME ]; then 
  rm -rf $ABS_DST/$OLD_ARCHIVE_INSTALL_FOLDERNAME
  #echo "[Debug] Delete old l_mlsl_$version_year.$version_num.$subversion_num folder!"
fi
} 

# MLSL
DST=`dirname $0`
#echo "[Debug] dirname: $0"
#echo "[Debug] DST value: $DST"
ABS_DST=`readlink -f $DST`
#echo "[Debug] ABS_DST value: $ABS_DST"

if [ -z $MLSL_ROOT ]; then
  #l_mlsl_p_2017.0.014 version has the different structure, so keep a seprate version
  CleanUpPreviousMLSL2017_0_014
  CleanUpPreviousMLSL 2017 1 016
  CleanUpPreviousMLSL 2017 2 018
  CleanUpPreviousMLSL 2018 0 003
fi

VERSION_MATCH=20181005
ARCHIVE_BASENAME=l_mlsl_2018.1.005.tgz
ARCHIVE_INSTALL_FOLDERNAME=l_mlsl_2018.1.005
#because the l_mlsl_2018.0.003.tgz will unpacked files.tar.gz and install.sh to the ARCHIVE_INSTALL_FOLDERNAME
#not unpacked to the DST folder (Different behavior against l_mlsl_p_2017.0.014.tgz)
ARCHIVE_INSTALL_FOLDERNAME_TEMP=l_mlsl_2018.1.005_temp
MLSL_CONTENT_DIR=`echo $ARCHIVE_BASENAME | rev | cut -d "." -f 2- | rev`
#echo "[Debug] MLSL_CONTENT_DIR value: $MLSL_CONTENT_DIR"
GITHUB_RELEASE_TAG=v2018.1-Preview

MLSLURL="https://github.com/01org/MLSL/releases/download/$GITHUB_RELEASE_TAG/$ARCHIVE_BASENAME"
#echo "[Debug] MLSL_ROOT value: $MLSL_ROOT"
VERSION_LINE=`GetVersionName $MLSL_ROOT`
#echo "[Debug] VERSION_LINE value: $VERSION_LINE"
# Check if MLSL_ROOT is set if positive then set one will be used..
if [ -z $MLSL_ROOT ] || [ $VERSION_LINE -lt $VERSION_MATCH ]; then
  # ..if MLSL_ROOT is not set then check if we have MLSL unpacked and installed in proper version
  FindLibrary $DST
  #echo "[Debug] LOCALMLSL value inside if: $LOCALMLSL"
  if [ $LOCALMLSL ]; then
    #in order to return value to calling script (Makefile,cmake), cannot print other info
    #echo "[Debug] Some verison of MLSL is unpacked and installed"
    MLSL_PREVIOUS_CONTENT_DIR=`echo $LOCALMLSL | rev | cut -d "/" -f 4- | cut -d "/" -f -1 | rev`
    #echo "[Debug] MLSL_PREVIOUS_CONTENT_DIR value: $MLSL_PREVIOUS_CONTENT_DIR"
    #echo "[Debug] DST/MLSL_PREVIOUS_CONTENT_DIR value: $DST/$MLSL_PREVIOUS_CONTENT_DIR"
    VERSION_LINE=`GetVersionName $DST/$MLSL_PREVIOUS_CONTENT_DIR`
  fi
  #echo "[Debug] VERSION_LINE value inside if: $VERSION_LINE"

  #if MLSL_ROOT is not set 
  if [ -z $MLSL_ROOT ]; then
    #if version is not given, or the version is lower than expected version 
    if [ $VERSION_LINE -lt $VERSION_MATCH ]; then
      #Then downloaded, unpacked and installed
      wget --no-check-certificate -P $DST $MLSLURL -O $DST/$ARCHIVE_BASENAME
      if [ ! -d $DST/$ARCHIVE_INSTALL_FOLDERNAME_TEMP ]; then
        mkdir $DST/$ARCHIVE_INSTALL_FOLDERNAME_TEMP
        #echo "[Debug] Create l_mlsl_2018.1.005_temp folder for unpacking!"
      fi
      tar -xzf $DST/$ARCHIVE_BASENAME -C $DST/$ARCHIVE_INSTALL_FOLDERNAME_TEMP
      #echo "[Debug] PWD value: $PWD"
      #install.sh did not support the relative path as the parameter
      bash $DST/$ARCHIVE_INSTALL_FOLDERNAME_TEMP/$ARCHIVE_INSTALL_FOLDERNAME/install.sh -s -d $ABS_DST/$ARCHIVE_INSTALL_FOLDERNAME
      rm -rf $DST/$ARCHIVE_INSTALL_FOLDERNAME_TEMP
      #echo "[Debug] Remove l_mlsl_2018.1.005_temp folder for unpacking!"
    fi
    #else: version is just our expected version, no need to donload again, but need to set the MLSL_ROOT
    #do not change the value of MLSL_ROOT if MLSL_ROOT is set, but version is not given
    FindLibrary $DST
    #echo "[Debug] LOCALMLSL value: $LOCALMLSL"
    #echo "[Debug] PWD value: $PWD"
    MLSL_ROOT=$PWD/`echo $LOCALMLSL | sed -e 's/intel64.*$//'`
  else
    #if MLSL_ROOT is set, but version is not given, or the version is lower than expected version
    #not to download our own version, and just use mlsl as the return value of LIBRARIES
    LIBRARIES="mlsl"
  fi
  #echo "[Debug] MLSL_ROOT value: $MLSL_ROOT"
fi

#The simplest implementation of LIBRARIES return value
LIBRARIES="mlsl"
#echo "[Debug] LIBRARIES value: $LIBRARIES"

# return value to calling script (Makefile,cmake)
echo $MLSL_ROOT $LIBRARIES
