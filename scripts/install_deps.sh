#!/bin/bash

os="centos"

sudo_passwd=""

# centos: yum; ubuntu: apt-get
install_command=""

command_prefix=""

root_dir=$(cd $(dirname $(dirname $0)); pwd)

function usage
{
    script_name=$0
    echo "Usage:"
    echo "  $script_name [--hostfile host_file]"
    echo ""
    echo "  Parameters:"
    echo "    host: host file includes list of nodes. Only used when you want to install dependencies for multinode"
}

function check_os
{
    echo "Check OS and the version..."

    centos_ver_file="/etc/centos-release"
    if [ -f $centos_ver_file ]; then
        os="centos"
        cat $centos_ver_file
    else
        ubuntu_ver_cmd="lsb_release"
        check_dependency  $ubuntu_ver_cmd
        if [ $? -eq 0 ]; then
            os="ubuntu"
            eval "$ubuntu_ver_cmd -a"
        else
            echo "Unknown OS. Exit."
            exit 1
        fi
    fi
    echo "Detected OS: $os."
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

function is_sudoer
{
    echo $sudo_passwd | sudo -S -E -v >/dev/null
    if [ $? -eq 1 ]; then
        echo "User $(whoami) is not sudoer, and cannot install dependencies."
        exit 1
    fi
}

function install_python_deps
{
    pip install --user --upgrade pip
    pushd $root_dir/python >/dev/null
    for req in $(cat requirements.txt) pydot;
    do
        pip install --user $req
    done
    popd >/dev/null
}
function install_deps
{
    echo "Install dependencies..."
    if [ "$os" == "centos" ]; then
        eval $package_installer clean all
        eval $package_installer install epel-release
        eval $package_installer groupinstall "Development Tools"
        eval $package_installer install python-devel boost boost-devel numpy \
            numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 \
            hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv \
            opencv-devel scipy
    elif [ "$os" == "ubuntu" ]; then
        eval $package_installer update
        eval $package_installer install build-essential
        eval $package_installer install pkg-config libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
        eval $package_installer install --no-install-recommends libboost-all-dev
        eval $package_installer install libgflags-dev libgoogle-glog-dev liblmdb-dev
        eval $package_installer install python-dev
        eval $package_installer install python-numpy python-scipy
        eval $package_installer install libopencv-dev
    fi

    eval $package_installer install cmake wget bc numactl python-pip
    install_python_deps
}

function install_python_deps_multinode
{
    ansible all -m shell -a "pip install --user --upgrade pip"
    pushd $root_dir/python >/dev/null
    for req in $(cat requirements.txt) pydot;
    do
        ansible all -m shell -a "pip install --user $req"
    done
    popd >/dev/null

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

    if [ "$os" == "centos" ]; then
        eval $package_installer install epel-release
        eval $package_installer clean all
        eval $package_installer groupinstall "Development Tools"
    elif [ "$os" == "ubuntu" ];  then
        eval $package_installer update
        eval $package_installer install build-essential
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

    if [ "$os" == "centos" ]; then
        ansible all -m shell -a "$package_installer install python-devel boost boost-devel numpy numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv opencv-devel scipy"
    elif [ "$os" == "ubuntu" ]; then
        ansible all -m shell -a "$package_installer install pkg-config libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler"
        ansible all -m shell -a "$package_installer install --no-install-recommends libboost-all-dev"
        ansible all -m shell -a "$package_installer install libgflags-dev libgoogle-glog-dev liblmdb-dev"
        ansible all -m shell -a "$package_installer install python-dev"
        ansible all -m shell -a "$package_installer install python-numpy python-scipy"
        ansible all -m shell -a "$package_installer install libopencv-dev"
    fi

    ansible all -m shell -a "$package_installer install mc cpuinfo htop tmux screen iftop iperf vim wget bc numactl cmake python-pip"
    ansible all -m shell -a "systemctl stop firewalld.service"
    install_python_deps_multinode
}

host_file=""
while [[ $# -ge 1 ]]
do
    key="$1"
    case $key in
        --hostfile)
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


check_os
if [ "$os" == "centos" ]; then
    install_command="yum"
elif [ "$os" == "ubuntu" ]; then
    install_command="apt-get"
fi

check_dependency $install_command
if [ $? -ne 0 ]; then
    echo "Please check if $os and $install_command is installed correctly."
    exit 1
fi

package_installer="$install_command -y "

# install dependencies
username=`whoami`
if [ "$username" != "root" ]; then
    read -s -p "Enter password for $username: " sudo_passwd
    command_prefix="echo $sudo_passwd | sudo -S -E " 
    package_installer="$command_prefix $install_command -y "
    is_sudoer
fi

echo "Install dependencies..."
if [ "$host_file" == "" ]; then
    install_deps
else
    install_deps_multinode $host_file
fi
