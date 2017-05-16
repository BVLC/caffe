#!/bin/bash
# File: caffe_install.sh
# Author: Jason D. Miller, hack-r.github.io
# Description: This script automatically installs and configures Caffe for Python on Ubuntu
# Comment: I wrote this for the modern version of Python because it has been in production over 8 years, but if you insist on using ancient Python, just change python3 to python and pip3 to pip

apt-get update
apt-get install -y git cmake gcc  libgflags-dev libgoogle-glog-dev liblmdb-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libboost1.55* python3-dev python3-pip python3
wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
tar zxvf glog-0.3.3.tar.gz
cd glog-0.3.3
./configure
make -j 4  && make install
cd ..
wget https://github.com/schuhschuh/gflags/archive/master.zip
unzip master.zip
cd gflags-master
mkdir build && cd build
export CXXFLAGS="-fPIC" 
cmake .. 
make -j 4 VERBOSE=1
make install
cd ../..
git clone https://github.com/LMDB/lmdb
cd lmdb/libraries/liblmdb
make && make install
cd ../../..
git clone  https://github.com/BVLC/caffe
cd caffe
for req in $(cat python/requirements.txt); do pip3 install $req; done
sed -i  's/python_version "2"/python_version "3"/g' CMakeLists.txt
cmake .
make -j 4
make install
