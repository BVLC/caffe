FROM ubuntu:16.04
MAINTAINER caffe-maint@googlegroups.com

#ENV http_proxy proxy:port
#ENV https_proxy proxy:port

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	    cpio \
        build-essential \
        cmake \
        git \
        wget \
        ssh \
        openssh-server \
        numactl \
        vim \
        net-tools \
        iputils-ping \
        screen \
        libmlx4-1 libmlx5-1 ibutils  rdmacm-utils libibverbs1 ibverbs-utils perftest infiniband-diags \
        openmpi-bin libopenmpi-dev \
        ufw \
        iptables \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/intel/caffe.git . && \
    pip install --upgrade pip && \
    for req in $(cat python/requirements.txt) pydot; do pip --no-cache-dir install $req; done && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 -DUSE_MLSL=1 -DCMAKE_BUILD_TYPE=Release .. && \
    make all -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /opt/caffe

RUN mkdir /var/run/sshd && \
    echo 'root:intelcaffe@123' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/;s/Port 22/Port 10010/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 10010
RUN ssh-keygen -t rsa -A && \
    mkdir ~/.ssh && \
    touch ~/.ssh/config && \
    echo "Host *" > ~/.ssh/config && \
    echo "Port 10010" >> ~/.ssh/config && \
    echo "StrictHostKeyChecking no" >> ~/.ssh/config
CMD ["/usr/sbin/sshd", "-D"]
