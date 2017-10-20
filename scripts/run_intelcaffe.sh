#!/bin/bash

benchmark_mode="none"

# by default, run intel caffe on single node
numnodes=1


# time/train/resume_train
mode="train"

# it's assigned by detect_cpu
cpu_model="skx"

# a list of nodes
host_file=""

# network parameters
network="opa"
tcp_netmask=""

# specify number of MLSL ep servers in command
num_mlsl_servers=-1

# parameters for caffe time
iteration=0
model_file=""
# parameters for resuming training
snapshot=""
# parameters for training
solver_file=""

# specify engine for running caffe
engine="MKLDNN"

#default numa node if needed
numanode=0

result_dir=""
debug="off"
mpibench_bin="IMB-MPI1"
mpibench_param="allreduce"

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--host host_file] [--solver solver_file]"
    echo "               [--network opa/tcp] [--netmask tcp_netmask] [--debug on/off]"
    echo "               [--mode train/resume_train/time/none] [--benchmark all/qperf/mpi/none]"
    echo "               [--iteration iter] [--model_file deploy.prototxt]"
    echo "               [--snapshot snapshot.caffemodel]"
    echo "               [--num_mlsl_servers num_mlsl_servers]"
    echo "               [--output output_folder]"
    echo "               [--mpibench_bin mpibench_bin]"
    echo "               [--mpibench_param mpibench_param]"
    echo ""
    echo "  Parameters:"
    echo "    host: host file includes list of nodes. Only used when you're running multinodes mode"
    echo "    solver: need to be specified a solver file if mode is train/resume_train"
    echo "    network: opa(default), tcp"
    echo "    netmask: only used if network is tcp"
    echo "    debug: off(default). MLSL debug information is outputed if it's on"
    echo "    mode: train(default), resume_train, time, none(not to run caffe test)"
    echo "    benchmark: none(disabled by default). Includes qperf, all-reduce performance"
    echo "      Dependency: user needs to install qperf, Intel MPI library (including IMB-MPI1);"
    echo "                  and add them in system path."
    echo "    iteration and model_file: only used if mode is time (caffe time)"
    echo "    snapshot: only used if mode is resume_train"
    echo "    num_mlsl_servers: number of MLSL ep servers"
    echo "    output_folder: output folder for storing results"
    echo "    mpibench_bin: IMB-MPI1 (default). relative path of binary of mpi benchmark."
    echo "    mpibench_param: allreduce (default). parameter of mpi benchmark."
}

declare -a cpu_list=("Intel Xeon E5-26xx (Broadwell)" "Intel Xeon Phi 72xx (Knights Landing)" 
                     "Intel Xeon Platinum 8180 (Skylake)" "Intel Xeon 6148 (Skylake)")

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
        cpu_model="unknown"
        echo "CPU model :$model_string is unknown."
        echo "    Use default settings, which may not be the optimal one."
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
        echo "Cache mode."
        # cache mode, use numa node 0
        numanode=0
    else
        echo "Flat mode."
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


function init_mpi_envs
{
    if [ ${numnodes} -eq 1 ]; then
        return
    fi

    # IMPI configuration
    if [ "$network" == "opa" ]; then
        export I_MPI_FABRICS=tmi
        export I_MPI_TMI_PROVIDER=psm2
        if [ "$cpu_model" == "knl" ];  then
            # PSM2 configuration
            export PSM2_MQ_RNDV_HFI_WINDOW=4194304 #2097152 # to workaround PSM2 bug in IFS 10.2 and 10.3
            export PSM2_MQ_EAGER_SDMA_SZ=65536
            export PSM2_MQ_RNDV_HFI_THRESH=200000
        fi

        export PSM2_IDENTIFY=1 # for debug
    elif [ "$network" == "tcp" ]; then
        export I_MPI_FABRICS=tcp
        export I_MPI_TCP_NETMASK=$tcp_netmask
    else
        echo "Invalid network: $network"
        exit 1
    fi

    export I_MPI_FALLBACK=0
    export I_MPI_DEBUG=6
}


function clear_shm
{
    clear_command="rm -rf /dev/shm/*"
    check_shm_command="df -h | grep shm"

    # TODO: check if 40G is the minimum shm size?
    min_shm_size=40
    shm_unit="G"

    for node in "${nodenames[@]}"
    do
        ssh ${node} "$clear_command"
        shm_line=`ssh ${node} "$check_shm_command"`
        shm_string=`echo $shm_line | awk -F ' ' '{print $(NF-2)}'`
        unit="${shm_string:(-1)}"
        shm_size=${shm_string::-1}
        if [ "$unit" == "$shm_unit" ] && [ $shm_size -ge ${min_shm_size} ]; then
            continue
        else
            echo "Warning: /dev/shm free size = ${shm_size}${unit}, on node: ${node}."
            echo "       Better to larger than ${min_shm_size}${shm_unit}."
            echo "       Please clean or enlarge the partition."
        fi
    done
}

function kill_zombie_processes
{
    kill_command="for process in ep_server caffe mpiexec.hydra; do for i in \$(ps -e | grep -w \$process | awk -F ' ' '{print \$1}'); do kill -9 \$i; echo \"\$process \$i killed.\"; done done"
    for node in "${nodenames[@]}"
    do
        ssh ${node} "$kill_command"
    done
}

function clear_envs
{
    clear_shm
    kill_zombie_processes
}

function set_mlsl_vars
{
    if [ ${num_mlsl_servers} -eq -1 ]; then
        if [ ${numnodes} -eq 1 ]; then
            numservers=0
        else
            if [ "${cpu_model}" == "bdw" ] || [ "${cpu_model}" == "skx" ]; then
                numservers=2
            else
                numservers=4
            fi
        fi
    else
        numservers=$((num_mlsl_servers))
    fi

    echo "MLSL_NUM_SERVERS: $numservers"
    export MLSL_NUM_SERVERS=${numservers}

    if [ ${numservers} -gt 0 ]; then
        if [ "${cpu_model}" == "bdw" ] || [ "${cpu_model}" == "skx" ]; then
            listep=6,7,8,9
        else
            listep=6,7,8,9,10,11,12,13
        fi
        export MLSL_SERVER_AFFINITY="${listep}"
        echo "MLSL_SERVER_AFFINITY: ${listep}"
    fi

    # MLSL configuration
    if [ "$debug" == "on" ]; then
        export MLSL_LOG_LEVEL=3
    else
        export MLSL_LOG_LEVEL=0
    fi
}

function set_env_vars
{
    set_mlsl_vars
    init_mpi_envs

    ppncpu=1
    threadspercore=1

    cores=`lscpu | grep "Core(s) per socket:" | awk '{print $4}'`
    sockets=`lscpu | grep "Socket(s)" | awk  '{print $2}'`
    maxcores=$((cores*sockets))

    numthreads=$(((maxcores-numservers)*threadspercore))
    numthreads_per_proc=$((numthreads/ppncpu))

    export OMP_NUM_THREADS=${numthreads_per_proc}

    # OMP configuration
    # threadspercore=1
    affinitystr="proclist=[0-5,$((5+numservers+1))-$((maxcores-1))],granularity=thread,explicit"
    export KMP_HW_SUBSET=1t
    export KMP_AFFINITY=$affinitystr
}

function execute_command
{
    local xeonbin_=$1
    local result_dir_=$2

    if [ "${cpu_model}" == "bdw" ]; then
        exec_command="$xeonbin_"
    elif [ "${cpu_model}" == "skx" ]; then
        exec_command="numactl -l $xeonbin_"
    else
        exec_command="numactl --preferred=$numanode $xeonbin_"
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

    clear_envs

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

    xeonbin="$mpibench_bin $mpibench_param"

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
            set_env_vars
            run_mpi_bench
        fi
    fi
}

function run_caffe
{
    echo "Run caffe with ${numnodes} nodes..."

    if [ ${mode} == "time" ]; then
        xeonbin="$caffe_bin time --iterations $iteration --model $model_file  -engine=$engine"
    else
        xeonbin="$caffe_bin train --solver $solver_file -engine=$engine"
        if [ ${mode} == "resume_train" ]; then
            xeonbin+=" --snapshot=${snapshot}"
        fi
    fi

    set_env_vars
    execute_command "$xeonbin" $result_dir
}


if [ $# -le 1 ]; then
    usage
    exit 0
fi

root_dir=$(cd $(dirname $(dirname $0)); pwd)
result_dir=${root_dir}/"result-`date +%Y%m%d%H%M%S`"
while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
        --solver)
            solver_file="$2"
            shift
            ;;
        --host)
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
echo ""
echo "Settings:"
echo "    Host file: $host_file"
echo "    Running mode: $mode"
echo "    Benchmark: $benchmark_mode"
echo "    Debug option: $debug"
echo "    Engine: $engine"
echo "    Number of MLSL servers: $num_mlsl_servers"
echo "        -1: selected automatically according to CPU model."
echo "            BDW/SKX: 2, KNL: 4"


if [ "$mode" == "train" ] || [ "$mode" == "resume_train" ]; then
    if [ "$solver_file" == "" ]; then
        echo "Error: solver file is NOT specified."
        exit 1
    fi
    if [ ! -f $solver_file ]; then
        echo "Error: solver file does NOT exist."
        exit 1
    fi

    echo "    Solver file: $solver_file"

    if [ "$mode" == "resume_train" ]; then
        if [ "$snapshot" == "" ]; then
            echo "Error: snapshot is NOT specified."
            exit 1
        fi
        if [ ! -f $snapshot ]; then
            echo "Error: snapshot file does NOT exist."
            exit 1
        fi
        echo "    Snapshot for resuming train: $snapshot"
    fi
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
    nodenames=( `cat $host_file | sort | uniq ` )
    if [ ${#nodenames[@]} -eq 0 ]; then
        echo "Error: empty host file! Exit."
        exit 0
    fi
    numnodes=${#nodenames[@]}
fi
echo "    Number of nodes: $numnodes"

detect_cpu

set_numa_node

if [ ! -d $result_dir ]; then
    echo "Create result directory: $result_dir"
    mkdir -p $result_dir
fi

if [ "${benchmark_mode}" != "none" ]; then
    run_benchmark
fi

if [ "${mode}" != "none" ]; then
    caffe_bin="./build/tools/caffe"
    check_dependency $caffe_bin
    if [ $? -ne 0 ]; then
        echo "Exit."
        exit 0
    fi

    run_caffe
fi

echo "Result folder: $result_dir"
