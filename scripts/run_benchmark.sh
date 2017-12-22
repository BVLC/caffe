#!/bin/bash

# model path
model_path="models/intel_optimized_models"

# network topology, support alexnet, googlenet, googlenet v2, resnet50
# if you set it as 'all', then it will benchmark all supported topologies.
topology=""
topology_list="alexnet googlenet googlenet_v2 resnet_50 all"
declare -a model_list=("alexnet" "googlenet" "googlenet_v2" "resnet_50")

# it's assigned by detect_cpu
cpu_model="skx"

# flag used to mark if we have detected which cpu model we're using
unknown_cpu=0

# specify default engine for running caffe benchmarks
engine=""

# default support single node
numnodes=1

# intelcaffe_log_file obtain outputs of 'run_intelcaffe'
intelcaffe_log_file=""

# specific script used to run intelcaffe 
caffe_run_script="./scripts/run_intelcaffe.sh"

# path of caffe binary
caffe_bin=""

# iterations to run benchmark
iterations=100

# hostfile needed to run multinodes mode benchmark
host_file=""

# network parameters
network="opa"
tcp_netmask=""


data_source="dummy"

function usage
{
    script_name=$0
    echo "Usage:"
    echo "   $script_name --topology network_topology [--hostfile host_file] [--network opa/tcp] [--netmask tcp_netmask]"
    echo "                [--data_source lmdb/dummy] [--caffe_bin caffe_bin_path] [--engine engine]"
    echo ""
    echo "   Parameters:"
    echo "     topology: network topology used to benchmark, support alexnet, googlenet, googlenet_v2, resnet_50"
    echo "               , by specifying it as 'all', we run all supported topologies."
    echo "     hostfile: host_file needed in multinodes mode, should contain list of nodes ips or hostnames"
    echo "     network: opa(default), tcp, used in multinodes mode to specify the network type"
    echo "     netmask: only used if network is tcp, set as the net work card name within your network"
    echo "     data_source: dummy(default), lmdb. data source for neural network."
    echo "     caffe_bin_path: specify path of caffe binary."
    echo "     engine: empty value(default), MKL2017 and MKLDNN. Default empty value"
    echo "             is to use default engine in Intel Caffe."
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

function calculate_numnodes
{
    if [[ $host_file != "" ]]; then
        host_list=(`cat $host_file | sort | uniq`)
        numnodes=${#host_list[@]}
        if [ $numnodes -eq 0 ]; then
            echo "Error: empty host list. Exit."
            exit 1
        fi
    fi
    echo "Number of nodes: $numnodes"
}

function detect_cpu
{
    # detect cpu model
    model_string=`lscpu | grep "Model name" | awk -F ':' '{print $2}'`
    if [[ $model_string == *"Phi"* ]]; then
        if [[ $model_string =~ 72(1|3|5|9)0 ]]; then
            cpu_model="knl"
        elif [[ $model_string == *"72"* ]]; then
            cpu_model="knm"
        else
            unknown_cpu=1
            echo "Can't detect which cpu model currently using, will use default settings, which may not be the optimal one."
        fi
    else
        model_num=`echo $model_string | awk '{print $4}'`
        if [[ $model_num =~ ^[8|6|5|4|3]1 ]]; then
            cpu_model="skx"
        elif [[ $model_num =~ ^E5-[4|2|1]6|^E7-[8|4]8|^E3-12|^D[-]?15 ]]; then
            cpu_model="bdw"
        else
            unknown_cpu=1
            echo "Can't detect which cpu model currently using, will use default settings, which may not be the optimal one."
        fi
    fi
}

function run_specific_model
{
    if [ $numnodes -eq 1 ]; then
        if [ "$data_source" == "dummy" ]; then
            model_file="models/intel_optimized_models/${model}/${cpu_model}/train_val_dummydata.prototxt"
        else
            model_file="models/intel_optimized_models/${model}/${cpu_model}/train_val.prototxt"
        fi
        exec_command="${caffe_run_script} --model_file ${model_file} --mode time --iteration ${iterations} --benchmark none"
    else
        if [ "$data_source" == "dummy" ]; then
            solver_file="models/intel_optimized_models/${model}/${cpu_model}/solver_dummydata.prototxt"
        else
            solver_file="models/intel_optimized_models/${model}/${cpu_model}/solver.prototxt"
        fi

        exec_command="${caffe_run_script} --hostfile $host_file --solver $solver_file --network $network --benchmark none"
        if [ $network == "tcp" ]; then
            exec_command+=" --netmask $tcp_netmask"
        fi
    fi 
    
    if [ "$engine" != "" ]; then
        exec_command+=" --engine $engine"
    fi
    if [ "$caffe_bin" != "" ]; then
        exec_command+=" --caffe_bin $caffe_bin"
    fi

    # Result file to save detailed run intelcaffe results
    if [ $unknown_cpu -eq 0 ]; then
        result_log_file="result-${cpu_model}-${model}-`date +%Y%m%d%H%M%S`.log"
    else
        result_log_file="result-unknown-${model}-`date +%Y%m%d%H%M%S`.log"
    fi
    $exec_command 2>&1 | tee  $result_log_file
    obtain_intelcaffe_log $result_log_file
    calculate_images_per_second $intelcaffe_log_file 
}

function obtain_intelcaffe_log
{
    echo "Result_log_file : $1"
    if [ -f $1 ]; then
       result_dir_line=`cat $1 | grep "Result folder:"`
       if [[ result_dir_line = "" ]]; then
           echo "Couldn't find result folder within file $1"
           exit 1
       fi
       result_dir=`echo $result_dir_line | awk -F ' ' '{print $(NF)}'`
       if [ $unknown_cpu -eq 0 ]; then
           caffe_log_file="outputCluster-${cpu_model}-${numnodes}.txt"
       else
           caffe_log_file="outputCluster-unknown-${numnodes}.txt"
       fi
       intelcaffe_log_file="${result_dir}/${caffe_log_file}"
    else
       echo "Couldn't see result log file $result_log_file"
       exit 1
    fi
}

function obtain_average_fwd_bwd_time
{
    result_file=$1
    if [ ! -f $result_file ]; then
        echo "Error: result file $result_file does not exist..."
        exit 1
    fi

    if [ $numnodes -eq 1 ]; then
        average_time_line=`cat $result_file | grep "Average Forward-Backward"`
        if [ "$average_time_line" = "" ]; then
            echo "running intelcaffe failed, please check logs under: $result_file"
            exit 1
        fi
        average_time=`echo $average_time_line | awk -F ' ' '{print $(NF-1)}'`
    else
        start_iteration=100
        iteration_num=100
        total_time=0
        deltaTimeList=`cat $result_file | grep "DELTA TIME" | tail -n "+${start_iteration}" | head -n ${iteration_num} | awk '{print $(NF-1)}'`
        if [ "$deltaTimeList" = "" ]; then
            echo "running intelcaffe failed, please check logs under: $result_file"
            exit 1
        fi
        
        for delta_time in ${deltaTimeList}
        do
            iteration_time=`echo "$delta_time" | bc`
            total_time=`echo "$total_time+$iteration_time" | bc`
        done

        average_time=`echo "$total_time*1000/$iteration_num" | bc`
    fi
    echo "average time: ${average_time}"
}

function obtain_batch_size
{
    log_file=$1
    if [ ! -f $log_file ]; then
        echo "Error: log file $log_file does not exist..."
        exit 1
    fi
    batch_size=`cat $log_file | grep shape | sed -n "3, 1p" | awk '{print $(NF-4)}'`
    batch_size_opt=`cat $log_file | grep SetMinibatchSize | sed -n "1, 1p" | awk '{print $(NF)}'`
    if [[ $batch_size_opt != "" ]]; then
        batch_size=$batch_size_opt
    fi
    echo "batch size: $batch_size"
}

function calculate_images_per_second 
{
    obtain_batch_size $1
    obtain_average_fwd_bwd_time $1
    if [ $numnodes -eq 1 ]; then
        speed=`echo "$batch_size*1000/$average_time" | bc`
    else
        speed=`echo "$batch_size*$numnodes*1000/$average_time" | bc`
    fi
    echo "benchmark speed : $speed images/sec"
}


function run_benchmark
{
    detect_cpu
    calculate_numnodes
    echo "Cpu model : $model_string"
    if [[ $topology = "all" ]]; then
       for ((i=0; i<${#model_list[@]}; i++))
       do
          echo "--${model_list[$i]}"
          model=${model_list[$i]}
          run_specific_model
       done
    else
       model=$topology
       run_specific_model
    fi 
}

function check_parameters
{
    if [[ $topology = "" ]]; then
        echo "Error: topology is not specified."
        usage
        exit 1
    fi
    
    if [[ $host_file != "" ]]; then
        if [ "$network" = "tcp" -a "$tcp_netmask" = "" ]; then
            echo "Error: need to specify tcp network's netmask"
            usage
            exit 1
        fi
    fi
    is_supported_topology
}

if [[ $# -le 1 ]]; then
    usage
    exit 0
fi

while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in 
        --topology)
            topology="$2"
            shift
            ;;
        --hostfile)
            host_file="$2"
            shift
            ;;
        --network)
            network="$2"
            shift
            ;;
        --netmask)
            tcp_netmask="$2"
            shift
            ;;
        --engine)
            engine="$2"
            shift
            ;;
        --data_source)
            data_source="$2"
            shift
            ;;
        --caffe_bin)
            caffe_bin=$2
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

check_parameters

run_benchmark
