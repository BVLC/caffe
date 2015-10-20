# Makefile for all third party libraries 

# NVidia CNMEM, https://github.com/NVIDIA/cnmem.git

.PHONY: cnmem

install: build
	cp $(CNMEM_BUILD_DIR)/*.so* $(CAFFE_BUILD_DIR)/lib

build : cnmem

CNMEM_BUILD_DIR=./cnmem/build
cnmem:
	$(MAKE) -C $(CNMEM_BUILD_DIR) DEBUG=$(DEBUG)



