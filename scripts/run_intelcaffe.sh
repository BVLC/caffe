#!/bin/bash

benchmark_mode="none"

# by default, run intel caffe on single node
numnodes=1


# time/train
mode="train"

# it's assigned by detect_cpu
cpu_model=""

# a list of nodes
host_file=""

# network parameters
network="opa"
tcp_netmask=""

# specify number of MLSL ep servers in command
num_mlsl_servers=-1

debug="off"

# parameters for caffe time
iteration=0
model_file=""
# parameters for resuming training
snapshot=""
# parameters for training
solver_file=""

# weights for finetuning
weight_file=""

# number of OpenMP threads
num_omp_threads=0

# specify engine for running caffe
engine=""

#default numa node if needed
numanode=0

# pin internal threads to 2 CPU cores for reading data
internal_thread_pin="on"

result_dir=""

mpibench_bin="IMB-MPI1"
mpibench_param="allreduce"

script_dir=$(dirname $0)

caffe_bin=""

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--hostfile host_file] [--solver solver_file]"
    echo "               [--weights weight_file] [--num_omp_threads num_omp_threads]"
    echo "               [--network opa/tcp] [--netmask tcp_netmask] [--debug on/off]"
    echo "               [--mode train/time/none] [--benchmark all/qperf/mpi/none]"
    echo "               [--iteration iter] [--model_file deploy.prototxt]"
    echo "               [--snapshot snapshot.caffemodel]"
    echo "               [--num_mlsl_servers num_mlsl_servers]"
    echo "               [--internal_thread_pin on/off]"
    echo "               [--output output_folder]"
    echo "               [--mpibench_bin mpibench_bin]"
    echo "               [--mpibench_param mpibench_param]"
    echo "               [--caffe_bin  caffe_binary_path]"
    echo "               [--cpu cpu_model]"
    echo ""
    echo "  Parameters:"
    echo "    hostfile: host file includes list of nodes. Only used if you're running with multinode"
    echo "    solver: need to be specified a solver file if mode is train"
    echo "    weight_file: weight file for finetuning"
    echo "    num_omp_threads: number of OpenMP threads"
    echo "    network: opa(default), tcp"
    echo "    netmask: only used if network is tcp"
    echo "    debug: off(default). MLSL debug information is outputed if it's on"
    echo "    mode: train(default), time, none(not to run caffe test)"
    echo "    benchmark: none(disabled by default). Includes qperf, all-reduce performance"
    echo "      Dependency: user needs to install qperf, Intel MPI library (including IMB-MPI1);"
    echo "                  and add them in system path."
    echo "    iteration and model_file: only used if mode is time (caffe time)"
    echo "    snapshot: it's specified if train is resumed"
    echo "    num_mlsl_servers: number of MLSL ep servers"
    echo "    internal_thread_pin: on(default). pin internal threads to 2 CPU cores for reading data."
    echo "    output_folder: output folder for storing results"
    echo "    mpibench_bin: IMB-MPI1 (default). relative path of binary of mpi benchmark."
    echo "    mpibench_param: allreduce (default). parameter of mpi benchmark."
    echo "    caffe_binary_path: path of caffe binary."
    echo "    cpu_model: specify cpu model and use the optimal settings if the CPU is not"
    echo "               included in supported list. Value: bdw, knl, skx and knm."
    echo "               bdw - Broadwell, knl - Knights Landing, skx - Skylake,"
    echo "               knm - Knights Mill."
    echo ""
}

declare -a cpu_list=("Intel Xeon E7-88/48xx, E5-46/26/16xx, E3-12xx, D15/D-15 (Broadwell)"
                     "Intel Xeon Phi 7210/30/50/90 (Knights Landing)" 
                     "Intel Xeon Platinum 81/61/51/41/31xx (Skylake)")

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
            cpu_model="unknown"
            echo "CPU model :$model_string is unknown."
            echo "    Use default settings, which may not be the optimal one."
        fi
    else
        model_num=`echo $model_string | awk '{print $4}'`
        if [[ $model_num =~ ^[8|6|5|4|3]1 ]]; then
            cpu_model="skx"
        elif [[ $model_num =~ ^E5-[4|2|1]6|^E7-[8|4]8|^E3-12|^D[-]?15 ]]; then
            cpu_model="bdw"
        else
            cpu_model="unknown"
            echo "CPU model :$model_string is unknown."
            echo "    Use default settings, which may not be the optimal one."
        fi
    fi
}

function set_numa_node
{
    check_dependency numactl
    if [ $? -ne 0 ]; then
        return
    fi

    # detect numa mode: cache and flat mode for KNL
    numa_node=($(numactl -H | grep "available" | awk -F ' ' '{print $2}'))
    if [ $numa_node -eq 1 ]; then
        echo "    NUMA configuration: Cache mode."
        # cache mode, use numa node 0
        numanode=0
    else
        echo "    NUMA configuration: Flat mode."
        numanode=1
    fi
}


function check_dependency
{
    dep=$1
    which $dep >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: cannot find $dep"
        return 1
    fi
    return 0
}


function execute_command
{
    local xeonbin_=$1
    local result_dir_=$2

    if [ "${cpu_model}" == "skx" ]; then
        exec_command="numactl -l $xeonbin_"
    elif [ "${cpu_model}" == "knl" ] || [ "${cpu_model}" == "knm" ]; then
        exec_command="numactl --preferred=$numanode $xeonbin_"
    else
        exec_command="$xeonbin_"
    fi

    if [ ${numnodes} -gt 1 ]; then
        # Produce the configuration file for mpiexec. 
        # Each line of the config file contains a # host, environment, binary name.
        cfile_=$result_dir_/nodeconfig-${cpu_model}-${numnodes}.txt
        rm -f $cfile_

        for node in "${nodenames[@]}"
        do
            echo "-host ${node} -n $ppncpu $exec_command" >> $cfile_
        done
    fi
    log_file=$result_dir_/outputCluster-${cpu_model}-${numnodes}.txt

    sensors_bin="sensors"
    check_dependency $sensors_bin
    has_sensors=$?
    if [ $has_sensors -eq 0 ]; then
        sensor_log_file=$result_dir_/sensors-${cpu_model}-${numnodes}-start.log
        $sensors_bin >$sensor_log_file
    fi

    if [ ${numnodes} -eq 1 ]; then
        time GLOG_minloglevel=0 $exec_command 2>&1 | tee ${log_file}
    else
        exec_command="-l -configfile $cfile_"
        time GLOG_minloglevel=0 mpiexec.hydra $exec_command 2>&1 | tee ${log_file}
    fi

    if [ $has_sensors -eq 0 ]; then
        sensor_log_file=$result_dir_/sensors-${cpu_model}-${numnodes}-end.log
        $sensors_bin >$sensor_log_file
    fi
}

function run_qperf_bench
{
    qperf_bin="qperf"
    check_dependency $qperf_bin
    if [ $? -ne 0 ]; then
        echo "Skip qperf benchmark."
        return
    fi

    # measure bandwidth and latency
    qperf_result_log="qperf_bench_result.log"
    rm -f $qperf_result_log

    server_node=""
    port=1234567
    qperf_param="-lp $port -oo msg_size:1024:512M:*2 -vu tcp_bw tcp_lat"

    for ((i=0; i<numnodes-1; i++))
    do
        server_node=${nodenames[$i]}
        echo "Run qperf server on ${server_node}..." | tee -a $qperf_result_log
        ssh -f $server_node "$qperf_bin -lp $port" >> $qperf_result_log
        echo >>$qperf_result_log

        for ((j=i+1; j<numnodes; j++))
        do
            client_node=${nodenames[$j]}
            echo "Run qperf client on ${client_node}..." | tee -a $qperf_result_log
            qperf_command="$qperf_bin $server_node $qperf_param"
            if [ ${j} == ${numnodes} ]; then
                qperf_command+=" quit"
            fi
            echo "ssh $client_node $qperf_command" | tee -a $qperf_result_log
            ssh $client_node "$qperf_command" | tee -a $qperf_result_log
            echo >>$qperf_result_log
        done
    done

    mv $qperf_result_log $result_dir/
}

function run_mpi_bench
{
    # MPI benchmark
    check_dependency $mpibench_bin
    if [ $? -ne 0 ]; then
        echo "Skip MPI benchmark..."
        return
    fi

    mpi_iter=10
    max_msglog=29
    xeonbin="$mpibench_bin $mpibench_param"
    if [ "$mpibench_bin" == "IMB-MPI1" ] || [ "$mpibench_bin" == "IMB-NBC" ]; then
        xeonbin+=" -msglog $max_msglog -iter $mpi_iter -iter_policy off"
    fi

    mpibench_bin_bname=`basename $mpibench_bin`

    if [ "${benchmark_mode}" == "all" ]; then
        declare -a adjust_values=(1 2 3 5 7 8 9 0)
        declare -a collective_values=('tmi' 'none')
    else
        declare -a adjust_values=(0)
        declare -a collective_values=('none')
    fi

    echo "Start mpi bench..."
    for ((i=0; i<${#adjust_values[@]}; i++))
    do
        for ((j=0; j<${#collective_values[@]}; j++))
        do
            if [ ${adjust_values[$i]} -eq 0 ]; then
                unset I_MPI_ADJUST_ALLREDUCE
            else
                export I_MPI_ADJUST_ALLREDUCE=${adjust_values[$i]}
            fi

            if [ "${collective_values[$j]}" == "none" ]; then
                unset I_MPI_COLLECTIVE_DEFAULTS
            else
                export I_MPI_COLLECTIVE_DEFAULTS=${collective_values[$j]}
            fi
            echo "iteration $i, ${j}..."
            echo "I_MPI_ADJUST_ALLREDUCE=$I_MPI_ADJUST_ALLREDUCE"
            echo "I_MPI_COLLECTIVE_DEFAULTS=$I_MPI_COLLECTIVE_DEFAULTS"

            test_result_dir=$result_dir/mpibench-${mpibench_bin_bname}-${mpibench_param}-${adjust_values[$i]}-${collective_values[$j]}
            mkdir -p $test_result_dir
            execute_command "$xeonbin" $test_result_dir
        done
    done

    # TODO: analyze the report and select the best algorithm and setting
    unset I_MPI_COLLECTIVE_DEFAULTS
    unset I_MPI_ADJUST_ALLREDUCE

    echo "Finished."
}

function run_benchmark
{
    echo "Run benchmark with ${numnodes} nodes..."
    if [ $numnodes -gt 1 ]; then
        if [ "$benchmark_mode" == "all" ] || [ "$benchmark_mode" == "qperf" ]; then
            run_qperf_bench
        fi

        if [ "$benchmark_mode" == "all" ] || [ "$benchmark_mode" == "mpi" ]; then
            run_mpi_bench
        fi
    fi
}

function run_caffe
{
    echo "Run caffe with ${numnodes} nodes..."

    if [ ${mode} == "time" ]; then
        xeonbin="$caffe_bin time --iterations $iteration --model $model_file"
    else
        xeonbin="$caffe_bin train --solver $solver_file"
        if [ "${snapshot}" != "" ]; then
            xeonbin+=" --snapshot=${snapshot}"
        fi
        if [ "${weight_file}" != "" ]; then
            xeonbin+=" --weights ${weight_file}"
        fi
    fi

    if [ "${engine}" != "" ]; then
        xeonbin+=" --engine=$engine"
    fi

    execute_command "$xeonbin" $result_dir
}

function test_ssh_connection
{
    host_file_=$1
    if [ "$host_file_" != "" ]; then
        host_list=( `cat $host_file_ | sort -V | uniq ` )
        for host in ${host_list[@]}
        do
            hostname=`ssh $host "hostname"`
            # prompt user to input password and no password should be
            # needed in the following commands
            ssh $hostname "ls" >/dev/null
        done
    fi
}

function get_model_fname
{
    solver_file_=$1
    model_file_=$(grep -w "net:" $solver_file_ | head -n 1 | awk -F ':' '{print $2}' | sed 's/\"//g' | sed 's/\r//g')
    echo "$(echo $model_file_)"
}

function check_lmdb_files
{
    model_file_=$1
    
    is_missing_lmdb=0
    lmdb_dirs=($(grep -w "source:" $model_file_ | sed 's/^ *//g' | grep -v "^#" | awk -F ' ' '{print $(NF)}' | sed 's/\"//g' | sed 's/\r//g'))
    for lmdb_dir in "${lmdb_dirs[@]}"
    do
        echo "    LMDB data source: $lmdb_dir"
        if [ ! -d "$lmdb_dir" ]; then
            echo "Error: LMDB data source doesn't exist ($lmdb_dir)."
            let is_missing_lmdb=1
        fi
    done

    if [ $is_missing_lmdb -eq 1 ]; then
        echo ""
        echo "Please follow the steps to create imagenet LMDB:"
        echo "    1. Please download images from image-net.org website."
        echo "    2. Execute script to download auxiliary data for training:"
        echo "           ./data/ilsvrc12/get_ilsvrc_aux.sh"
        echo "    3. update parameters in examples/imagenet/create_imagenet.sh"
        echo "           TRAIN_DATA_ROOT and VALUE_DATA_ROOT: path of training and validation images from image-net.org"
        echo "           RESIZE=true/false: resize to 256x256 if true"
        echo "           ENCODE=true/false: LMDB is compressed if true"
        echo "    4. Execute script to create lmdb for imagenet"
        echo "           ./examples/imagenet/create_imagenet.sh"
        echo ""
        echo "See details in Intel Caffe github wiki:"
        echo "    https://github.com/intel/caffe/wiki/How-to-create-Imagenet-LMDB"

        exit -1
    fi 
}

if [ $# -le 1 ]; then
    usage
    exit 0
fi

root_dir=$(cd $(dirname $(dirname $0)); pwd)
result_dir=${root_dir}/"result-`date +%Y%m%d%H%M%S`"
while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --solver)
            solver_file="$2"
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
        --debug)
            debug="$2"
            shift
            ;;
        --num_mlsl_servers)
            num_mlsl_servers=$2
            shift
            ;;
        --mode)
            mode=$2
            shift
            ;;
        --iteration)
            iteration=$2
            shift
            ;;
        --model_file)
            model_file=$2
            shift
            ;;
        --snapshot)
            snapshot=$2
            shift
            ;;
        --weights)
            weight_file=$2
            shift
            ;;
        --num_omp_threads)
            num_omp_threads=$2
            shift
            ;;
        --internal_thread_pin)
            internal_thread_pin=$2
            shift
            ;;
        --engine)
            engine=$2
            shift
            ;;
        --benchmark)
            benchmark_mode=$2
            shift
            ;;
        --output)
            result_dir=$2
            shift
            ;;
        --mpibench_bin)
            mpibench_bin=$2
            shift
            ;;
        --mpibench_param)
            mpibench_param=$2
            shift
            ;;
        --help)
            usage
            ;;
        --engine)
            engine=$2
            shift
            ;;
        --caffe_bin)
            caffe_bin=$2
            shift
            ;;
        --cpu)
            cpu_model=$2
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

echo ""
echo "CPUs with optimal settings:"
for ((i=0; i<${#cpu_list[@]}; i++))
do
    echo "    ${cpu_list[$i]}"
done

# if cpu model is not specified in command, detect cpu model by "lscpu" command
if [ "$cpu_model" == "" ]; then
    detect_cpu
fi

echo ""
echo "Settings:"
echo "    CPU: $cpu_model"
echo "    Host file: $host_file"
echo "    Running mode: $mode"
echo "    Benchmark: $benchmark_mode"
echo "    Debug option: $debug"
echo "    Engine: $engine"
echo "    Number of MLSL servers: $num_mlsl_servers"
echo "        -1: selected automatically according to CPU model."
echo "            BDW/SKX: 2, KNL: 4"


if [ "$mode" == "train" ]; then
    if [ "$solver_file" == "" ]; then
        echo "Error: solver file is NOT specified."
        exit 1
    fi
    if [ ! -f $solver_file ]; then
        echo "Error: solver file does NOT exist."
        exit 1
    fi

    echo "    Solver file: $solver_file"

    if [ "$snapshot" != "" ]; then
        if [ ! -f $snapshot ]; then
            echo "Error: snapshot file does NOT exist."
            exit 1
        fi
        echo "    Snapshot for resuming train: $snapshot"
    fi
    model_file=$(get_model_fname $solver_file)
fi

if [ "$mode" == "time" ]; then
    if [ "$model_file" == "" ]; then
        echo "Error: model file is NOT specified."
        exit 1
    fi
    if [ ! -f $model_file ]; then
        echo "Error: model file does NOT exist."
        exit 1
    fi

    if [ $iteration -le 0 ]; then
        echo "Error: iteration ($iteration) <= 0."
        exit 1
    fi        
    echo "    Iteration for running caffe time: $iteration"
    echo "    Model file for running caffe time: $model_file"
fi

# check source data exists
if [ "$model_file" != "" ]; then
    grep "backend" $model_file | grep -i "LMDB"  >/dev/null
    if [ $? -eq 0 ]; then
        check_lmdb_files $model_file
    fi
fi

echo "    Network: $network"
if [ "$network" == "tcp" ]; then
    if  [ "$tcp_netmask" == "" ]; then
        echo "Error: TCP netmask is NOT specified."
        exit 0
    fi
    echo "    Netmask for TCP network: $tcp_netmask"
fi

# Names to configfile, binary (executable) files #
# Add check for host_file's existence to support single node
if [[ $host_file != "" ]]; then
    nodenames=( `cat $host_file | sort -V | uniq ` )
    if [ ${#nodenames[@]} -eq 0 ]; then
        echo "Error: empty host file! Exit."
        exit 0
    fi
    numnodes=${#nodenames[@]}
fi

# test connection between nodes via ssh
test_ssh_connection $host_file

set_numa_node

if [ ! -d $result_dir ]; then
    echo "Create result directory: $result_dir"
    mkdir -p $result_dir
fi

env_params="--cpu $cpu_model --debug $debug --network $network --num_mlsl_servers $num_mlsl_servers"
if [ "$network" == "tcp" ]; then
    env_params+=" --netmask $tcp_netmask"
fi
if [ "$host_file" != "" ]; then
    env_params+=" --hostfile $host_file"
fi
if [ ${num_omp_threads} -ne 0 ]; then
    env_params+=" --num_omp_threads ${num_omp_threads}"
fi

source ${script_dir}/set_env.sh $env_params

if [ "${benchmark_mode}" != "none" ]; then
    run_benchmark
fi

if [ "${mode}" != "none" ]; then
    if [ "$caffe_bin" == "" ]; then
      caffe_bin="${root_dir}/build/tools/caffe"
    fi

    check_dependency $caffe_bin
    if [ $? -ne 0 ]; then
        echo "Exit."
        exit 0
    fi

    run_caffe
fi

echo "Result folder: $result_dir"
