FROM centos:7
MAINTAINER caffe-maint@googlegroups.com

#ENV http_proxy proxy:port
#ENV https_proxy proxy:port

RUN rpm -iUvh https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-11.noarch.rpm && \
    yum upgrade -y && \
    yum install -y \
        bzip2 \
        numactl \
        redhat-rpm-config \
        tar \
        findutils \
        make \
        gcc-c++ \
        cmake \
        git \
        vim \
        wget \
        ssh \
        openssh.x86_64 \
        openssh-server.x86_64 \
        openssh-clients.x86_64 \
        initscripts  \
        net-tools \
        ufw \
        iptables \
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
        libjpeg-turbo-devel && \
    yum clean all

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

# If you want to deploy container in HOST network mode, Install SSH service and config it to Non-standard Port. Or you needn’t below rows.
RUN mkdir /var/run/sshd && \
    echo "root:intelcaffe@123" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/;s/#Port 22/Port 10086/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 10086
RUN ssh-keygen -t rsa -A && \
    mkdir ~/.ssh && \
    touch ~/.ssh/config && \
    echo "Host *" > ~/.ssh/config && \
    echo "Port 10086" >> ~/.ssh/config && \
    echo "StrictHostKeyChecking no" >> ~/.ssh/config
CMD ["/usr/sbin/sshd", "-D"]
