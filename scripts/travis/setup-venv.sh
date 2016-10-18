#!/bin/bash
# setup a Python virtualenv
# (must come after install-deps)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

VENV_DIR=${1:-~/venv}

# setup our own virtualenv
if $WITH_PYTHON3; then
    PYTHON_EXE='/usr/bin/python3'
else
    PYTHON_EXE='/usr/bin/python2'
fi

# use --system-site-packages so that Python will use deb packages
virtualenv $VENV_DIR -p $PYTHON_EXE --system-site-packages
