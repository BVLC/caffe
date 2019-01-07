# Caffe (GAN edition)

This is a modified version of caffe to adapt GAN training. 

## Build

Currently, only `Makefile` method is supported, and the steps is the same as original caffe. `CMake` or other method is not tested. Only ubuntu platform is tested, but other platform is supposed to work.

## Run

Currently, DCGAN on Cifar10 and MNIST is tested. After you have prepared the dataset, run the `run.sh` script at `examples` directory. Both GPU and CPU mode are supported.

On some machines, you should make the log directory in previous, or there may be write permission errors. E.g. the `log/mnist_gan` directory for MNIST GAN experiment.



## Functionality

- DCGAN training on MNIST, Cifar10, or any custom data.

- Any traditional generator or discriminator architecture: not including new layers like Spectral Normalization.

- Minmax loss function for GAN.

## Future functionality

- Auxiliary Classifier.

- cycleGAN.

NOTICE:

- Any gradient penalty will not be supported: WGAN-CP, DRAGAN. Because it is too hard to compute the gradient of gradient in Caffe.

## Customization

This GAN edition only conducted minimal modification of GAN. Modified files is listed at follows, if you do care about it:

(`A` for added file, `M` for modified file)

- A: `tools/caffe_gan.cpp`. The training interface of GAN.

- A: `examples/cifar_gan`, `examples/mnist_gan`, `models/gan/`. Some examples of GAN.

- A: `src/caffe/layers/randvec_layer.cpp`, `include/layers/randvec_layer.hpp`. The random noise layer for generator.

- M: `src/caffe/proto/caffe.proto`. Register `RandVecLayer`.

- A: `src/caffe/gan_solver.cpp`, `include/caffe/gan_solver.hpp`. Add a solver for general GAN training.

- M: `src/caffe/net.cpp`, `include/caffe/net.hpp`. Modified the interface of `Net` class to expose more `Forward` and `Backward` functionality.

In summary, no aggressive modification is done.

