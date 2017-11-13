FROM centos:7
MAINTAINER caffe-maint@googlegroups.com

#ENV http_proxy proxy:port
#ENV https_proxy proxy:port

RUN rpm -iUvh http://download.fedoraproject.org/pub/epel/7/x86_64/e/epel-release-7-10.noarch.rpm

RUN yum upgrade -y

RUN yum install -y \
        redhat-rpm-config \
        tar \
        findutils \
        make \
        gcc-c++ \
        cmake \
        git \
        wget \
        ssh \
        atlas-devel \
        boost-devel \
        gflags-devel \
        glog-devel \
        hdf5-devel \
        leveldb-devel \
        lmdb-devel \
        opencv-devel \
        protobuf-devel \
        snappy-devel \
        protobuf-compiler \
        freetype-devel \
        libpng-devel \
        python-devel \
        python-numpy \
        python-pip \
        python-scipy \
        gcc-gfortran \
        libjpeg-turbo-devel

RUN yum clean all 
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/intel/caffe.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip --no-cache-dir install $req; done && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 -DCMAKE_BUILD_TYPE=Release .. && \
    make all -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
