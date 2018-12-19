FROM ubuntu:16.04
LABEL maintainer caffe-maint@googlegroups.com

# Build as
# sudo docker build . -f docker/gpu/Dockerfile -t l4b/caffe:ocl

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
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
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools \
        python3-scipy \
	python3-skimage \
	liblmdb-dev \
	python3-lmdb \
	doxygen \
	libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=opencl
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git .

# Following line commented out due to https://github.com/pypa/pip/issues/5599
# RUN pip install --upgrade pip

RUN cd python && for req in $(cat requirements.txt) wheel pydot; do pip3 install $req; done && cd ..

# OpenCL-specific stuff here
RUN apt-get update && apt-get install --no-install-recommends -y \
	ocl-icd-libopencl1 \
	ocl-icd-opencl-dev \
	libviennacl-dev \
	clinfo && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir build && cd build && \
    cmake -DUSE_OPENCL=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT:$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace

# test with:
# sudo docker run -it caffe/caffe:ocl caffe device_query -gpu 0
