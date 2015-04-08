add-apt-repository ppa:xorg-edgers/ppa
apt-get update
apt-get install `apt-cache search nvidia | grep 343 | awk '{print $1}'`

wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda-repo-ubuntu1404-6-5-prod_6.5-19_amd64.deb

dpkg -i cuda-repo-ubuntu1404-6-5-prod_6.5-19_amd64.deb
apt-get update
apt-get install cuda
apt-get install nvidia-cuda-toolkit
apt-get install libatlas-base-dev

# setting up caffe
apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev

apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler

sudo apt-get install gcc-4.7 g++-4.7 gcc-4.7-multilib g++-4.7-multilib
sudo apt-get install python-sklearn python-protobuf python-skimage

# Compiling Caffe
# https://github.com/BVLC/caffe/issues/337
# CUSTOM_CXX := g++-4.7

echo -e "\n\n===================================================="
echo "Add the following line to your .bashrc"
echo "    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
