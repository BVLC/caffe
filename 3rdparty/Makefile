# Makefile for all third party libraries 

# NVidia CNMEM, https://github.com/NVIDIA/cnmem.git

.PHONY: cnmem

all:
	@echo No targets to build! 3rd party target must be defined.

CNMEM_BUILD_DIR=./cnmem/build
cnmem:
	-rm -fr $(CNMEM_BUILD_DIR)
	mkdir -p $(CNMEM_BUILD_DIR)
	(cd $(CNMEM_BUILD_DIR) && cmake ..)
	$(MAKE) -C $(CNMEM_BUILD_DIR) DEBUG=$(DEBUG)
	cp $(CNMEM_BUILD_DIR)/*.so* $(INSTALLDIR)/lib
