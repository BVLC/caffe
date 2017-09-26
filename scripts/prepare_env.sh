#!/bin/bash

os="centos"

username=`whoami`
if [ "$username" != "root" ];
then
    package_installer="sudo -E"
fi

# centos: yum; ubuntu: apt-get
package_installer+=" yum -y"

function install_deps
{
    echo "Install dependencies..."
    if [ "$os" == "centos" ]; then
        $package_installer clean all
    	$package_installer upgrade
        $package_installer install epel-release
        $package_installer groupinstall "Development Tools"
    fi
    
    $package_installer install python-devel boost boost-devel cmake numpy \
        numpy-devel gflags gflags-devel glog glog-devel protobuf protobuf-devel hdf5 \
        hdf5-devel lmdb lmdb-devel leveldb leveldb-devel snappy-devel opencv \
        opencv-devel wget bc numactl
}

function check_os
{
    echo "Check OS and the version..."
}


function checkout_source
{
    echo "Checkout source code of Intel Caffe..."
    git clone https://github.com/intel/caffe.git
    if [ $? -eq 128 ]; then
        echo "Error during checking out source code. Please set proxy as below:"
        echo "    export https_proxy=https://username:password@proxy.com:port"
    fi
}
function build_caffe
{
    echo "Build Intel Caffe..."
    cp Makefile.config.example Makefile.config
    make -j 8
}

function is_sudoer
{
    sudo -v >/dev/null
    if [ $? -eq 1 ]; then
        echo "User $(whoami) is not sudoer, and cannot install dependencies."
        return 1
    fi
    return 0
}

check_os
if [ "$os" == "ubuntu" ]; then
    package_installer="apt-get"
fi

is_sudoer
if [ $? -eq 0 ]; then
    install_deps
fi


build_caffe

echo "Done."
