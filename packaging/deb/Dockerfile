FROM caffe-nv-debuild-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        dh-make \
        devscripts \
        equivs \
        lintian \
    && rm -rf /var/lib/apt/lists/*

ENV DEBFULLNAME "NVIDIA CORPORATION"
ENV DEBEMAIL "digits@nvidia.com"
ENV DEB_BUILD_OPTIONS "nocheck parallel=4"

ARG UPSTREAM_VERSION
ARG DEBIAN_VERSION
ARG CUDA_VERSION

WORKDIR /build
COPY tarball/* .
RUN tar -xf *.orig.tar.gz
WORKDIR /build/caffe-nv
RUN dh_make -y -s -c bsd -d -t `pwd`/packaging/deb/templates \
        -f ../*.orig.tar.gz -p caffe-nv_${UPSTREAM_VERSION} \
    && dch --create --package caffe-nv -v $DEBIAN_VERSION "v${DEBIAN_VERSION}" \
    && dch -r ""
RUN apt-get update \
    && echo y | mk-build-deps -i -r debian/control \
    && rm -rf /var/lib/apt/lists/*
RUN debuild -e CUDA_VERSION --no-tgz-check --no-lintian -i -uc -us -b \
    && lintian ../*.changes
RUN mkdir -p /dist \
    && cp ../* /dist/ || true
