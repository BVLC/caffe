#!/bin/bash
# configure the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_CMAKE ; then
  source $BASEDIR/configure-make.sh
else
  source $BASEDIR/configure-cmake.sh
fi
