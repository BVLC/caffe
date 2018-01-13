#!/bin/bash

# by default, run intel caffe on single node
numnodes=1

debug="off"

# it's assigned by detect_cpu
cpu_model="skx"

# number of OpenMP threads
num_omp_threads=0

nodenames=""
# a list of nodes
host_file=""

# network parameters
network="opa"
tcp_netmask=""

# specify number of MLSL ep servers in command
num_mlsl_servers=-1

numservers=0

# pin internal threads to 2 CPU cores for reading data
internal_thread_pin="on"

function init_mpi_envs
{
    if [ ${numnodes} -eq 1 ]; then
        return
    fi

    # IMPI configuration
    if [ "$network" == "opa" ]; then
        export I_MPI_FABRICS=tmi
        export I_MPI_TMI_PROVIDER=psm2
        if [ "$cpu_model" == "knl" ] || [ "$cpu_model" == "knm" ];  then
            # PSM2 configuration
            export PSM2_MQ_RNDV_HFI_WINDOW=2097152 # to workaround PSM2 bug in IFS 10.2 and 10.3
            export HFI_NO_CPUAFFINITY=1
            export I_MPI_DYNAMIC_CONNECTION=0
            export I_MPI_SCALABLE_OPTIMIZATION=0
            export I_MPI_PIN_MODE=lib 
            export I_MPI_PIN_DOMAIN=node
        fi
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
        if [ "${node}" == "" ]; then
            echo "Warning: empty node name."
            continue
        fi

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
    if [ ${numnodes} -eq 1 ]; then
        return
    fi

    if [ -z $MLSL_ROOT ]; then
        # use built-in mlsl if nothing is specified in ini
        mlslvars_sh=`find external/mlsl/ -name mlslvars.sh`
        source $mlslvars_sh
    fi

    if [ ${num_mlsl_servers} -eq -1 ]; then
        if [ "${cpu_model}" == "knl" ] || [ "${cpu_model}" == "knm" ]; then
            numservers=4
        else
            numservers=2
        fi
    else
        numservers=$((num_mlsl_servers))
    fi

    echo "MLSL_NUM_SERVERS: $numservers"
    export MLSL_NUM_SERVERS=${numservers}

    if [ ${numservers} -gt 0 ]; then
        if [ "${cpu_model}" == "knl" ] || [ "${cpu_model}" == "knm" ]; then
            listep=6,7,8,9
        else
            listep=6,7
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

function set_openmp_envs
{
    ppncpu=1
    threadspercore=1

    cpus=`lscpu | grep "^CPU(s):" | awk '{print $2}'`
    cores=`lscpu | grep "Core(s) per socket:" | awk '{print $4}'`
    sockets=`lscpu | grep "Socket(s)" | awk  '{print $2}'`
    maxcores=$((cores*sockets))

    if [ "$internal_thread_pin" == "on" ]; then
        export INTERNAL_THREADS_PIN=$((cpus-2)),$((cpus-1))
        echo "Pin internal threads to: $INTERNAL_THREADS_PIN"
    fi
    numthreads=$(((maxcores-numservers)*threadspercore))
    numthreads_per_proc=$((numthreads/ppncpu))

    # OMP configuration
    # For multinodes 
    if [ ${numnodes} -gt 1 ]; then
        reserved_cores=0
        if [ ${num_omp_threads} -ne 0 ]; then
            if [ $numthreads_per_proc -lt $num_omp_threads ]; then
                echo "Too large number of OpenMP thread: $num_omp_threads"
                echo "    should be less than or equal to $numthreads_per_proc"
                exit 1
            fi
            let reserved_cores=numthreads_per_proc-num_omp_threads
            echo "Reserve number of cores: $reserved_cores"
            let numthreads_per_proc=${num_omp_threads}
        fi

        export OMP_NUM_THREADS=${numthreads_per_proc}
        export KMP_HW_SUBSET=1t
        affinitystr="proclist=[0-5,$((5+numservers+reserved_cores+1))-$((maxcores-1))],granularity=thread,explicit"
        export KMP_AFFINITY=$affinitystr
    else
        # For single node only set for KNM
        if [ "${cpu_model}" == "knm" ]; then 
            export KMP_BLOCKTIME=10000000
            export MKL_ENABLE_INSTRUCTIONS=AVX512_MIC_E1
            export OMP_NUM_THREADS=${numthreads_per_proc}
            affinitystr="compact,1,0,granularity=fine"
            export KMP_AFFINITY=$affinitystr
        fi
    fi

    echo "Number of OpenMP threads: ${numthreads_per_proc}"
}

function set_env_vars
{
    set_mlsl_vars
    init_mpi_envs

    # depend on numservers set in set_mlsl_vars function
    set_openmp_envs
}

while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
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
        --cpu)
            cpu_model=$2
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
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
    shift
done

# Names to configfile, binary (executable) files #
# Add check for host_file's existence to support single node
if [[ $host_file != "" ]]; then
    nodenames=( `cat $host_file | sort | uniq ` )
    if [ ${#nodenames[@]} -eq 0 ]; then
        echo "Error: empty host file! Exit."
        exit 0
    fi
else
    nodenames=(`hostname`)
fi
numnodes=${#nodenames[@]}
echo "    Number of nodes: $numnodes"

clear_envs
set_env_vars
