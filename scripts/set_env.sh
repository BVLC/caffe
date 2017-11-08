#!/bin/bash

# by default, run intel caffe on single node
numnodes=1

debug="off"

# it's assigned by detect_cpu
cpu_model="skx"

nodenames=""
# a list of nodes
host_file=""

# network parameters
network="opa"
tcp_netmask=""

# specify number of MLSL ep servers in command
num_mlsl_servers=-1


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
            export PSM2_MQ_EAGER_SDMA_SZ=65536
            export PSM2_MQ_RNDV_HFI_THRESH=200000
            export HFI_NO_CPUAFFINITY=1
            export I_MPI_DYNAMIC_CONNECTION=0
            export I_MPI_SCALABLE_OPTIMIZATION=0
            export I_MPI_PIN_MODE=lib 
            export I_MPI_PIN_DOMAIN=node
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

function set_openmp_envs
{
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
    if [ "${cpu_model}" == "knl" ] || [ "${cpu_model}" == "knm" ]; then
        export KMP_BLOCKTIME=10000000
        export MKL_FAST_MEMORY_LIMIT=0
        if [ ${numnodes} -eq 1 ]; then
            affinitystr="compact,1,0,granularity=fine"
        fi
    fi
    export KMP_AFFINITY=$affinitystr
}

function set_env_vars
{
    set_mlsl_vars
    init_mpi_envs

    # depend on numservers set in set_mlsl_vars function
    set_openmp_envs
}

while [[ $# -gt 1 ]]
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
    numnodes=${#nodenames[@]}
fi
echo "    Number of nodes: $numnodes"

clear_envs
set_env_vars
