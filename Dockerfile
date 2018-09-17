FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
LABEL maintainer mail4mh@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        doxygen \
        git \
        graphviz \
        libavcodec-dev \
        libavformat-dev \
        libboost-all-dev \
        libfftw3-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgtk2.0-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libswscale-dev \
        libthrust-dev \
        pkg-config \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        python-setuptools \
        python-tk \
        unzip \
        vim \
        wget && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

# CMake 3.11
RUN cd /tmp && wget -q https://cmake.org/files/v3.11/cmake-3.11.4.tar.gz && \
    tar -xf cmake-3.11.4.tar.gz && \
    cd cmake-3.11.4 && ./configure --prefix=/usr && make -j"$(nproc)" && make install && \
    cd /tmp && rm -fr /tmp/cmake*

# HDF5 1.10.2
RUN cd /tmp && wget -q -O hdf5-1.10.2.tar.gz https://www.hdfgroup.org/package/source-gzip-2/?wpdmdl=11810 && \
    tar -xf hdf5-1.10.2.tar.gz && \
    cd hdf5-1.10.2 && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && make -j"$(nproc)" && make install && \
    cd /tmp && rm -fr /tmp/hdf5*

# GNU Scientific Library (GSL)
RUN cd /tmp && wget -q ftp://ftp.gnu.org/gnu/gsl/gsl-2.5.tar.gz && \
    tar -xf gsl-2.5.tar.gz && \
    cd gsl-2.5 && ./configure --prefix=/usr && make -j"$(nproc)" && make install && \
    cd /tmp && rm -fr /tmp/gsl*

ENV CAFFE_ROOT=/opt/caffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH

WORKDIR $CAFFE_ROOT

# Python packages
RUN pip install --upgrade pip
COPY python /opt/caffe/python
RUN cd python && pip install -r requirements.txt

COPY . /opt/caffe

RUN touch /opt/caffe/data/CMakeLists.txt

RUN ./build.sh

ENTRYPOINT caffe

WORKDIR /workspace

