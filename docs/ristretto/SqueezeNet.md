#Brewing a Fixed Point SqueezeNet

SqueezeNet [1] by Iandola et al. has the accuracy of AlexNet [2], but with over 50X fewer network parameters. This guide explains how to quantize SqueezeNet to fixed point, fine-tune the condensed network, and finally benchmark the net on the ImageNet validation data set.

In order to reproduce the following results, you first need to do these two steps:

* Download the SqueeNet parameters from [here](https://github.com/DeepScale/SqueezeNet) and put them into models/SqueezeNet folder.
* Do two modifications to the SqueezeNet prototxt file (models/SqueezeNet/train_val.prototxt): You need to adjust the path to your local ImageNet data for both *source* fields.

[1] Iandola, Forrest N., et al. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size](http://arxiv.org/abs/1602.07360). arXiv preprint (2016).

[2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. [Imagenet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). Advances in neural information processing systems. 2012.

##Quantization to Dynamic Fixed Point
*This guide assumes you previously installed Ristretto (make all) and that you run all commands from Caffe root.*

In a first step, we condense the 32-bit floating point network to dynamic fixed point. SqueezeNet performs well with 32 and 16-bit dynamic fixed point, however, we can reduce the bit-width further. There is a trade-off between parameter compression and network accuracy. The Ristretto tool can automatically find the appropriate bit-width for each part of the network:

    ./examples/ristretto/quantize_squeezenet.sh

This script will quantize the SqueezeNet model. You will see messages flying by as Ristretto tests the quantized model with different bit-widths. The final summary will look like this:

```
I0316 14:47:08.991994  5804 quantization.cpp:272] Network accuracy analysis for
I0316 14:47:08.992046  5804 quantization.cpp:273] Convolutional (CONV) and fully
I0316 14:47:08.992058  5804 quantization.cpp:274] connected (FC) layers.
I0316 14:47:08.992068  5804 quantization.cpp:275] Baseline 32bit float: 0.5768
I0316 14:47:08.992107  5804 quantization.cpp:276] Fixed point CONV weights: 
I0316 14:47:08.992120  5804 quantization.cpp:278] 16bit: 	0.557159
I0316 14:47:08.992135  5804 quantization.cpp:278] 8bit: 	0.555959
I0316 14:47:08.992151  5804 quantization.cpp:278] 4bit: 	0.00568
I0316 14:47:08.992164  5804 quantization.cpp:280] Fixed point FC weights: 
I0316 14:47:08.992174  5804 quantization.cpp:282] 16bit: 	0.5768
I0316 14:47:08.992188  5804 quantization.cpp:282] 8bit: 	0.5768
I0316 14:47:08.992202  5804 quantization.cpp:282] 4bit: 	0.5768
I0316 14:47:08.992215  5804 quantization.cpp:282] 2bit: 	0.5768
I0316 14:47:08.992229  5804 quantization.cpp:282] 1bit: 	0.5768
I0316 14:47:08.992244  5804 quantization.cpp:284] Fixed point layer outputs:
I0316 14:47:08.992252  5804 quantization.cpp:286] 16bit: 	0.57632
I0316 14:47:08.992269  5804 quantization.cpp:286] 8bit: 	0.57136
I0316 14:47:08.992285  5804 quantization.cpp:286] 4bit: 	0.0883603
I0316 14:47:08.992300  5804 quantization.cpp:288] Fixed point net:
I0316 14:47:08.992310  5804 quantization.cpp:289] 8bit CONV weights,
I0316 14:47:08.992319  5804 quantization.cpp:290] 1bit FC weights,
I0316 14:47:08.992328  5804 quantization.cpp:291] 8bit layer outputs:
I0316 15:22:00.895778  7547 quantization.cpp:292] Accuracy: 0.554379
```

The analysis shows that both the outputs and parameters of convolutional layers can be reduced to 8-bit with a top-1 accuracy drop of less than 2%. Since SqueezeNet contains no fully connected layers, the quantization results of this layer type can be ignored. Finally combining the quantization of all considered network parts yields an accuracy of 55.44% (compared to the baseline of 57.68%). In order to improve these results, we will fine-tune the network in the next step.

##Fine-tune Dynamic Fixed Point Parameters
The previous step quantized the 32-bit floating point SqueezeNet to 8-bit fixed point and generated the appropriate network description file (models/SqueezeNet/demo/quantized.prototxt). We can now fine-tune the condensed network to regain as much of its original accuracy as possible.

During fine-tuning, Ristretto will keep a set of high-precision weights. For each training batch, these 32-bit floating point weights are stochastically rounded to 8-bit fixed point. The 8-bit parameters are then used for the forward and backward propagation, and finally the weight update is applied to the high precision weights.

The fine-tuning procedure can be done with the traditional caffe-tool. Just start the following script:

    ./examples/ristretto/finetune_squeezenet.sh

After 1,200 fine-tuning iterations (~5h on a Tesla K-40 GPU) with batch size 32*32, our condensed SqueezeNet will have a top-1 validation accuracy of around 57.2%. The fine-tuned net parameters are located at models/SqueezeNet/demo/squeezenet_iter_1200.caffemodel. All in all, you successfully trimmed SqueezeNet to 8-bit dynamic fixed point, with an accuracy loss below 1%.

##Benchmark Dynamic Fixed Point SqueezeNet
In this step, you will benchmark an existing fixed point SqueezeNet which we fine-tuned for you. You can do the scoring even if you skipped the previous fine-tuning step. The model can be benchmarked with the traditional caffe-tool. All the tool needs is a network description file as well as the network parameters.

    ./examples/ristretto/benchmark_fixedpoint_squeezenet.sh

You should get a top-1 accuracy of 57.2%.
