#!/bin/bash

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--hostfile host_file] [--compiler icc/gcc]"
    echo ""
    echo "  Parameters:"
    echo "    host: host file includes list of nodes. Only used when you want to install dependencies for multinode"
    echo "    compiler: specify compiler to build intel caffe. default compiler is icc."
}


compiler="icc"
host_file=""
while [[ $# -gt 1 ]]
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
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

script_dir=$(cd $(dirname $0); pwd)


params=""
if [ "$host_file" != "" ] && [ -f $host_file ]; then
    params+=" --hostfile $host_file"
fi
$script_dir/install_deps.sh $params

echo "Build caffe..."
params="--compiler $compiler --rebuild "
if [ "$host_file" != "" ] && [ -f $host_file ]; then
    params+=" --multinode"
fi
$script_dir/build_intelcaffe.sh $params

echo "Done."
