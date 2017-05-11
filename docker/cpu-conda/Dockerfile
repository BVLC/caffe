FROM ubuntu:16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        wget \
        libgtk2.0-0 \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libpng12-dev \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh


ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
ENV CLONE_TAG=1.0

ARG CONDA=/opt/conda/bin/conda

RUN $CONDA update --yes conda

RUN $CONDA create -n condacaffe -y python=3.5

RUN $CONDA install -n condacaffe -c menpo -y opencv3

RUN /bin/bash -c "git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    source /opt/conda/bin/activate condacaffe && mkdir build && cd build && \
    cmake -DCPU_ONLY=1 -Dpython_version=3 .. && \
    make -j$(nproc) && \
    $CONDA install -y cython scikit-image ipython h5py nose pandas protobuf pyyaml jupyter && \
    cd ../python && sed -i -e 's/python-dateutil>=1.4,<2/python-dateutil>=2.0/g' requirements.txt && \
    for req in $(cat requirements.txt); do pip install $req; done"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:/opt/conda/bin:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
