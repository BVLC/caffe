FROM centos:centos7

# DEVELOPMENT BASE
RUN yum install -y epel-release
RUN yum install -y git wget unzip
RUN yum install -y make gcc-c++
RUN yum install -y python-devel python-pip && pip install --upgrade pip

# CMAKE
RUN cd /opt && wget -O cmake.tar.gz https://cmake.org/files/v3.5/cmake-3.5.0.tar.gz
RUN cd /opt && tar -xvzf cmake.tar.gz
RUN cd /opt/cmake-3.5.0 && ./configure && gmake -j 6 && gmake install

# LIBRARIES
RUN yum install -y freetype-devel libpng-devel libjpeg-devel qhull-devel boost-devel protobuf-devel
RUN yum install -y glog-devel gflags-devel hdf5-devel lmdb-devel leveldb-devel snappy-devel
RUN yum install -y atlas-devel lapack-devel openblas-devel opencv-devel

# CAFFE
ENV CAFFE_ROOT /opt/caffe
ENV CAFFE_RELEASE rc3

RUN cd $(dirname $CAFFE_ROOT) && wget -O caffe.tar.gz https://github.com/BVLC/caffe/archive/$CAFFE_RELEASE.tar.gz
RUN cd $(dirname $CAFFE_ROOT) && tar -xvzf caffe.tar.gz && mv caffe-$CAFFE_RELEASE $CAFFE_ROOT

WORKDIR $CAFFE_ROOT

RUN pip install numpy
RUN pip install -r python/requirements.txt
RUN pip install flask tornado # DEMO REQUIREMENTS

RUN mkdir build && cd build && cmake .. -DBLAS=Open
RUN cd build && make -j 4

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
