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

msg_priority="off"
msg_priority_threshold=""

mpi_iallreduce_algo=""

ppn=1

function init_mpi_envs
{
    if [ ${numnodes} -eq 1 ] && [ $ppn -eq 1 ]; then
        return
    fi

    # IMPI configuration
    if [ "$network" == "opa" ]; then
        export I_MPI_FABRICS=shm:tmi
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
        export I_MPI_FABRICS=shm:tcp
        export I_MPI_TCP_NETMASK=$tcp_netmask
    else
        echo "Invalid network: $network"
        exit 1
    fi

    export I_MPI_FALLBACK=0
    export I_MPI_DEBUG=6

    if [ "${mpi_iallreduce_algo}" != "" ]; then
        export I_MPI_ADJUST_IALLREDUCE=${mpi_iallreduce_algo}
    fi
    if [ ${ppn} -gt 1 ]; then
        maxcores=64
        domain_str="["
        for ((ppn_index=0; ppn_index<ppn; ppn_index++))
        do
            # limit of cpu masks: 10 (640 cores in maximum)
            declare -a masks=($(for i in {1..10}; do echo 0; done))

            start=$((ppn_index*cores_per_proc))
            end=$(((ppn_index+1)*cores_per_proc))

            mask_str="0x"
            for((cpu_index=start;cpu_index<end;cpu_index++))
            do
                echo $listep | grep -w $cpu_index >/dev/null
                if [ $? -ne 0 ]; then
                  index=$((cpu_index/maxcores))
                  offset=$((cpu_index%maxcores))
                  masks[$index]=$((masks[$index]+(1<<offset)))
                fi
            done
            for ((mask_index=${#masks[@]}-1; mask_index>=0; mask_index--))
            do
                mask_str+=`printf "%x" "${masks[$mask_index]}"`
            done
            domain_str+="$mask_str"
            if [ $ppn_index -ne $((ppn-1)) ]; then
                domain_str+=","
            fi
        done
        domain_str+="]"
        echo "I_MPI_PIN_DOMAIN=$domain_str"
        export I_MPI_PIN_DOMAIN=$domain_str
    fi
}


function clear_shm
{
    clear_command="rm -rf /dev/shm/*"
    check_shm_command="df -h /dev/shm | grep shm"

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
    if [ ${numnodes} -eq 1 ] && [ $ppn -eq 1 ]; then
        return
    fi

    if [ -z $MLSL_ROOT ]; then
        # use built-in mlsl if nothing is specified in ini
        mlslvars_sh=`find external/mlsl/ -name mlslvars.sh`
        if [ -f $mlslvars_sh ]; then
            source $mlslvars_sh
        fi
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
        # first cpu reserved for EP server
        first_cpu=6
        for ((ppn_index=0; ppn_index<${ppn}; ppn_index++))
        do
            # first cpu for current process
            first_cpu_proc=$((first_cpu+ppn_index*cores_per_proc))
            for ((srv_index=0; srv_index<${numservers}; srv_index++))
            do
                cpu_no=$((first_cpu_proc+srv_index))
                listep+="$cpu_no"
                if [ $ppn_index -ne $((ppn-1)) ] || [ $srv_index -ne $((numservers-1)) ]; then
                    listep+=","
                fi
            done
        done

        export MLSL_SERVER_AFFINITY="${listep}"
        echo "MLSL_SERVER_AFFINITY: ${listep}"
    fi

    # MLSL configuration
    if [ "$debug" == "on" ]; then
        export MLSL_LOG_LEVEL=3
    else
        export MLSL_LOG_LEVEL=1
    fi

    if [ "$msg_priority" == "on" ]; then
        echo "Enable priority queue."
        export MLSL_MSG_PRIORITY=1

        if [ "$msg_priority_threshold" != "" ]; then
            echo "Priority queue threshold: $msg_priority_threshold"
            export MLSL_MSG_PRIORITY_THRESHOLD=$msg_priority_threshold
        fi
    fi
}

function set_openmp_envs
{
    threadspercore=1

    if [ "$internal_thread_pin" == "on" ]; then
        internal_thread_str=""
        for ((ppn_index=0; ppn_index<ppn;ppn_index++))
        do
            last_thread_proc=$((total_cores*(num_ht-1)+total_cores*(ppn_index+1)/ppn-1))
            internal_thread_str+="$((last_thread_proc-1)),$last_thread_proc"
            if [ $ppn_index -ne $((ppn-1)) ]; then
                internal_thread_str+=","
            fi
        done

        export INTERNAL_THREADS_PIN=$internal_thread_str
        echo "Pin internal threads to: $INTERNAL_THREADS_PIN"
    fi

    numthreads_per_proc=$(((total_cores/ppn-numservers)*threadspercore))

    # OMP configuration
    # For multinodes 
    if [ ${numnodes} -gt 1 ] || [ $num_omp_threads -ne 0 ] || [ $ppn -gt 1 ]; then
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
        if [ $ppn -gt 1 ]; then
            affinitystr="granularity=fine,compact,1,0"
        else
            affinitystr="proclist=[0-5,$((5+numservers+reserved_cores+1))-$((total_cores-1))],granularity=thread,explicit"
        fi
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
        --msg_priority)
            msg_priority=$2
            shift
            ;;
        --msg_priority_threshold)
            msg_priority_threshold=$2
            shift
            ;;
        --mpi_iallreduce_algo)
            mpi_iallreduce_algo=$2
            shift
            ;;
        --ppn)
            ppn=$2
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


num_cpus=`lscpu | grep "^CPU(s):" | awk '{print $2}'`
cores_per_socket=`lscpu | grep "Core(s) per socket:" | awk '{print $4}'`
sockets=`lscpu | grep "Socket(s)" | awk  '{print $2}'`
total_cores=$((cores_per_socket*sockets))
# number of hyper threading
num_ht=$((num_cpus/total_cores))
cores_per_proc=$((total_cores/ppn))


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
