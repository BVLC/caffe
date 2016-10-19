# A makefile to build the docker images for caffe.
# Three caffe images can be built:
#   caffe:cpu-centos --> A CPU-only build of caffe in CentOS-based image.
#   caffe:cpu-ubuntu --> A CPU-only build of caffe in Ubuntu-based image.
#   caffe:gpu --> A GPU-enabled build using the latest CUDA and CUDNN versions.

DOCKER ?= docker

ifneq ($(http_proxy),)
DOCKER_BUILD_ARG += --build-arg http_proxy=$(http_proxy)
endif
ifneq ($(https_proxy),)
DOCKER_BUILD_ARG += --build-arg https_proxy=$(https_proxy)
endif


all: docker_files standalone

.PHONY: standalone devel

cpu_standalone: cpu-centos cpu-ubuntu
standalone: cpu_standalone gpu_standalone

cpu-centos: standalone/cpu-centos/Dockerfile
	$(DOCKER) build $(DOCKER_BUILD_ARG) -t caffe:cpu-centos standalone/cpu-centos

cpu-ubuntu: standalone/cpu-ubuntu/Dockerfile
	$(DOCKER) build $(DOCKER_BUILD_ARG) -t caffe:cpu-ubuntu standalone/cpu-ubuntu

gpu_standalone: standalone/gpu/Dockerfile
	$(DOCKER) build $(DOCKER_BUILD_ARG) -t caffe:gpu standalone/gpu


docker_files: standalone_files

standalone_files: standalone/cpu-centos/Dockerfile standalone/cpu-ubuntu/Dockerfile standalone/gpu/Dockerfile 

FROM_GPU = "nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04"
FROM_CPU = "ubuntu:14.04"
GPU_CMAKE_ARGS = -DUSE_CUDNN=1
CPU_CMAKE_ARGS = -DCPU_ONLY=1

# A make macro to select the CPU or GPU base image.
define from_image
$(if $(strip $(findstring gpu,$@)),$(FROM_GPU),$(FROM_CPU))
endef

# A make macro to select the CPU or GPU build args.
define build_args
$(if $(strip $(findstring gpu,$@)),$(GPU_CMAKE_ARGS),$(CPU_CMAKE_ARGS))
endef

# A make macro to construct the CPU or GPU Dockerfile from the template
define create_docker_file
	@echo creating $@
	@echo "FROM "$(from_image) > $@
	@cat $^ | sed 's/$${CMAKE_ARGS}/$(build_args)/' >> $@
endef


standalone/%/Dockerfile: templates/Dockerfile.template
	$(create_docker_file)

