---
layout: default
title: Caffe
---

# Performance, Hardware tips

To measure the performance of different Nvidia cards we use the reference imagenet model provided in Caffe.

## K40 Nvidia \*
 
### With ECC on

K40 ecc on max speed 26.7 secs / 20 training iterations (256*20 images), 101 secs / validation test (50000 images)
K40 ecc on default speed 31 secs / 20 training iterations (256*20 images), 117 secs / validation test (50000 images)

### With ECC off

K40 ecc off max speed 26.5 secs / 20 training iterations (256*20 images), 100 secs / validation test (50000 images)
K40 ecc off default speed 31 secs / 20 training iterations (256*20 images), 118 secs / validation test (50000 images)

### K40 Performance tip

To get the maximum performance of K40 NVidia one can adjust clock speed and dissable ecc (at your own risk).

To turn off ECC and reboot
	sudo nvidia-smi -e 0
Active permance flag
	sudo nvidia-smi -pm 1
and then set clocks speed
	sudo nvidia-smi -i 0 -ac 3004,875 


## Titan Nvidia \*

Titan 26.26 secs / 20 training iterations (256*20 images), 100 secs / validation test (50000 images)

## K20 Nvidia \*

Titan 36.0 secs / 20 training iterations (256*20 images), 133 secs / validation test (50000 images)

\* BVLC members are very gratefull to Nvidia for providing several GPU cards for conducting this research.