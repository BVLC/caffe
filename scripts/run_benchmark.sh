#!/bin/sh

# model path
model_path="models/intel_optimized_models"

# network topology, support alexnet, googlenet, googlenet v2, resnet50
# if you set it as 'all', then it will benchmark all supported topologies.
topology=""
topology_list="alexnet googlenet googlenet_v2 resnet_50 all"
declare -a model_list=("alexnet" "googlenet" "googlenet_v2" "resnet_50")

# it's assigned by detect_cpu
cpu_model="skx"

# specify default engine for running caffe benchmarks
engine="MKL2017"

# directory path to save results
result_dir=""

# specific script used to run intelcaffe 
caffe_bin="./scripts/run_intelcaffe.sh"

# Iterations to run benchmark
iterations=5

function usage
{
    script_name=$0
    echo "Usage:"
    echo "   $script_name --topology network_topology"
    echo ""
    echo "   Parameters:"
    echo "     topology: network topology used to benchmark, support alexnet, googlenet, googlenet_v2, resnet_50"
    echo "               , by specifying it as 'all', we run all supported topologies."
    echo ""
}

function is_supported_topology
{
    echo ""
    if [[ "$topology_list" =~ (^|[[:space:]])"$topology"($|[[:space:]]) ]]; then
        echo "Topology: ${topology}"
    else
        echo "$topology is not supported, please check the usage." 
        usage 
        exit 1
    fi
}

function detect_cpu
{
    # detect cpu model
    model_string=`lscpu | grep "Model name" | awk -F ':' '{print $2}'`
    if [[ $model_string == *"72"* ]]; then
        cpu_model="knl"
    elif [[ $model_string == *"8180"* ]]; then
        cpu_model="skx"
    elif [[ $model_string == *"6148"* ]]; then
        cpu_model="skx"
    elif [[ $model_string == *"E5-26"* ]]; then
        cpu_model="bdw"
    else
        echo "Will use default settings, which may not be the optimal one."
    fi
}

function run_specific_model
{
    model_file="models/intel_optimized_models/${model}/${cpu_model}/train_val_dummydata.prototxt"
    exec_command="${caffe_bin} --model_file ${model_file} --mode time --iteration ${iterations} --benchmark none"
    $exec_command
}

function run_benchmark
{
    echo "Cpu model : $model_string"
    if [[ $topology = "all" ]]; then
       for ((i=0; i<${#model_list[@]}; i++))
       do
          model=${model_list[$i]}
          run_specific_model
       done
    else
       model=$topology
       run_specific_model
    fi 
}

if [[ $# -le 1 ]]; then
    usage
    exit 0
fi

root_dir=$(cd $(dirname $(dirname $0)); pwd)
while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in 
        --topology)
            topology="$2"
            shift
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

# check parameters
if [[ $topology = "" ]]; then
    echo "Error: topology is not specified."
    exit 1
fi

# check if input topology is supported
is_supported_topology

detect_cpu

# start running benchmark
run_benchmark 
