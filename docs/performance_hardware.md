---
title: Performance and Hardware Configuration
---

# Performance and Hardware Configuration

To measure performance on different NVIDIA GPUs we use CaffeNet, the Caffe reference ImageNet model.

For training, each time point is 20 iterations/minibatches of 256 images for 5,120 images total. For testing, a 50,000 image validation set is classified.

**Acknowledgements**: BVLC members are very grateful to NVIDIA for providing several GPUs to conduct this research.

## NVIDIA K40

Performance is best with ECC off and boost clock enabled. While ECC makes a negligible difference in speed, disabling it frees ~1 GB of GPU memory.

Best settings with ECC off and maximum clock speed in standard Caffe:

* Training is 26.5 secs / 20 iterations (5,120 images)
* Testing is 100 secs / validation set (50,000 images)

Best settings with Caffe + [cuDNN acceleration](http://nvidia.com/cudnn):

* Training is 19.2 secs / 20 iterations (5,120 images)
* Testing is 60.7 secs / validation set (50,000 images)

Other settings:

* ECC on, max speed: training 26.7 secs / 20 iterations, test 101 secs / validation set
* ECC on, default speed: training 31 secs / 20 iterations, test 117 secs / validation set
* ECC off, default speed: training 31 secs / 20 iterations, test 118 secs / validation set

### K40 configuration tips

For maximum K40 performance, turn off ECC and boost the clock speed (at your own risk).

To turn off ECC, do

    sudo nvidia-smi -i 0 --ecc-config=0    # repeat with -i x for each GPU ID

then reboot.

Set the "persistence" mode of the GPU settings by

    sudo nvidia-smi -pm 1

and then set the clock speed with

    sudo nvidia-smi -i 0 -ac 3004,875    # repeat with -i x for each GPU ID

but note that this configuration resets across driver reloading / rebooting. Include these commands in a boot script to intialize these settings. For a simple fix, add these commands to `/etc/rc.local` (on Ubuntu).

## NVIDIA Titan

Training: 26.26 secs / 20 iterations (5,120 images).
Testing: 100 secs / validation set (50,000 images).

cuDNN Training: 20.25 secs / 20 iterations (5,120 images).
cuDNN Testing: 66.3 secs / validation set (50,000 images).


## NVIDIA K20

Training: 36.0 secs / 20 iterations (5,120 images).
Testing: 133 secs / validation set (50,000 images).

## NVIDIA GTX 770

Training: 33.0 secs / 20 iterations (5,120 images).
Testing: 129 secs / validation set (50,000 images).

cuDNN Training: 24.3 secs / 20 iterations (5,120 images).
cuDNN Testing: 104 secs / validation set (50,000 images).
