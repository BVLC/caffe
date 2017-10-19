#!/bin/bash

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--host host_file]"
    echo ""
    echo "  Parameters:"
    echo "    host: host file includes list of nodes. Only used when you want to install dependencies for multinode"
}

function check_os
{
    # echo "Check OS and the version..."
    echo "Only CentOS is supported."
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


sudo_passwd=""

function is_sudoer
{
    echo $sudo_passwd | sudo -S -E -v >/dev/null
    if [ $? -eq 1 ]; then
        echo "User $(whoami) is not sudoer, and cannot install dependencies."
	exit 1
    fi
}

# centos: yum; ubuntu: apt-get
os="centos"
install_command=""

check_os
if [ "$os" == "centos" ]; then
    install_command="yum"
    check_dependency $install_command
    if [ $? -ne 0 ]; then
        echo "Please check if CentOS and $install_command is installed correctly."
        exit 1
    fi
fi

package_installer="$install_command -y "
username=`whoami`
if [ "$username" != "root" ];
then
    read -s -p "Enter password for $username: " sudo_passwd
    is_sudoer
    package_installer="echo $sudo_passwd | sudo -S -E $install_command -y "
fi



function install_deps
{
    echo "Install dependencies..."
    if [ "$os" == "centos" ]; then
        eval $package_installer clean all
        eval $package_installer upgrade
        eval $package_installer install epel-release
        eval $package_installer groupinstall "Development Tools"
    fi

    eval $package_installer install python-devel boost boost-devel cmake numpy \
        numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 \
        hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv \
        opencv-devel wget bc numactl
}

function install_deps_multinode
{
    host_file=$1
    host_list=(`cat $host_file | sort | uniq`)

    host_cnt=${#host_list[@]}
    if [ $host_cnt -eq 0 ]; then
        echo "Error: empty host list. Exit."
        exit 1
    fi

    echo "Make sure you're executing command on host ${host_list[0]}"

    echo $sudo_passwd | sudo -S -E yum -y clean all

    if [ "$os" == "centos" ]; then
        eval $package_installer upgrade
        eval $package_installer install epel-release
        eval $package_installer clean all
        eval $package_installer groupinstall "Development Tools"
    fi

    eval $package_installer install ansible

    tmp_host_file=ansible_hosts.tmp
    ansible_host_file=/etc/ansible/hosts
    echo -e "[ourmaster]\n${host_list[0]}\n[ourcluster]\n" >$tmp_host_file
    for ((i=1; i<${#host_list[@]}; i++))
    do
        echo -e "${host_list[$i]}\n" >>$tmp_host_file
    done
    $command_prefix mv -f $tmp_host_file $ansible_host_file

    ssh-keygen -t rsa -q
    for host in ${host_list[@]}
    do
        ssh-copy-id -i ~/.ssh/id_rsa.pub $host
    done
    ansible ourcluster -m ping

    ansible all -m shell -a "$package_installer install python-devel boost boost-devel cmake numpy numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv opencv-devel"

    ansible all -m shell -a "$package_installer install mc cpuinfo htop tmux screen iftop iperf vim wget bc numactl"
    ansible all -m shell -a "systemctl stop firewalld.service"
}

function build_caffe
{
    is_multinode_=$1

    echo "Build Intel Caffe..."
    cp Makefile.config.example Makefile.config

    if [ $is_multinode_ -eq 1 ]; then
        echo "USE_MLSL := 1" >> Makefile.config
        echo "CAFFE_PER_LAYER_TIMINGS := 1" >> Makefile.config
	
        mlslvars_sh=`find external/mlsl/ -name mlslvars.sh`
        if [ -f $mlslvars_sh ]; then
            source $mlslvars_sh
        fi
    fi

    make -j 8
}

function sync_caffe_dir
{
  caffe_dir=`pwd`
  caffe_parent_dir=`dirname $caffe_dir`
  ansible ourcluster -m synchronize -a "src=$caffe_dir dest=$caffe_parent_dir"
}


host_file=""
while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
        --host)
            host_file="$2"
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


if [ $? -eq 0 ]; then
    if [ "$host_file" == "" ]; then
        install_deps
    else
        install_deps_multinode $host_file
    fi
fi

is_multinode=0
if [ "$host_file" != "" ]; then
    is_multinode=1
fi
build_caffe $is_multinode

if [ $is_multinode -eq 1 ]; then
  sync_caffe_dir
fi

echo "Done."
