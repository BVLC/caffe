#!/bin/bash

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--hostfile host_file] [--compiler icc/gcc] [--help] [--skip_install] [--skip_build]"
    echo ""
    echo "  Parameters:"
    echo "    host: host file includes list of nodes. Only used when you want to install dependencies for multinode"
    echo "    compiler: specify compiler to build Intel Caffe. default compiler is icc."
    echo "    help: print usage."
    echo "    skip_install: skip installing dependencies for Intel Caffe."
    echo "    skip_build: skip building Intel Caffe."
}

skip_install=0
skip_build=0
compiler="icc"
host_file=""
while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --hostfile)
            host_file="$2"
            shift
            ;;
        --compiler)
            compiler="$2"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        --skip_install)
            skip_install=1
            ;;
        --skip_build)
            skip_build=1
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

script_dir=$(cd $(dirname $0); pwd)

if [ $skip_install -eq 1 ]; then
    echo "Skip installing dependencies for Intel Caffe."
else
    params=""
    if [ "$host_file" != "" ] && [ -f $host_file ]; then
        params+=" --hostfile $host_file"
    fi
    $script_dir/install_deps.sh $params
fi

if [ $skip_build -eq 1 ]; then
    echo "Skip building Intel Caffe."
else
    echo "Build Caffe..."
    params="--compiler $compiler --rebuild "
    if [ "$host_file" != "" ] && [ -f $host_file ]; then
        params+=" --multinode --layer_timing"
    fi
    $script_dir/build_intelcaffe.sh $params
fi

echo "Done."
