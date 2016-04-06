#Ristretto Tool

The `ristretto` command line interface allows to automatically quantize a network. The tool finds the smallest possible bit-width representation, according to the user-defined maximum accuracy drop. Moreover, the tool generates the  protocol buffer definition file of the quantized net.

The tool is compiled to ./build/tools/ristretto. Run `ristretto` without any arguments for help.

##Example

The following command quantizes LeNet to Dynamic Fixed Point:

```
./.build/tools/ristretto quantize --model=examples/mnist/lenet_train_test.prototxt \
    --weights=examples/mnist/lenet_iter_10000.caffemodel \
    --model_quantized=examples/mnist/quantized.prototxt \
    --iterations=100 --gpu=0 --trimming_mode=fixed_point --error_margin=1
```

Given the error margin of 1%, LeNet can be quantized to 2-bit convolution kernels, 4-bit parameters in fully connected layers and 8-bit layer outputs:

```
I0331 10:36:48.470381  8047 quantization.cpp:442] ------------------------------
I0331 10:36:48.470438  8047 quantization.cpp:443] Network accuracy analysis for
I0331 10:36:48.470453  8047 quantization.cpp:444] Convolutional (CONV) and fully
I0331 10:36:48.470465  8047 quantization.cpp:445] connected (FC) layers.
I0331 10:36:48.470479  8047 quantization.cpp:446] Baseline 32bit float: 0.9915
I0331 10:36:48.470499  8047 quantization.cpp:447] Fixed point CONV weights: 
I0331 10:36:48.470512  8047 quantization.cpp:449] 16bit: 	0.9915
I0331 10:36:48.470533  8047 quantization.cpp:449] 8bit: 	0.9915
I0331 10:36:48.470553  8047 quantization.cpp:449] 4bit: 	0.9909
I0331 10:36:48.470573  8047 quantization.cpp:449] 2bit: 	0.9853
I0331 10:36:48.470593  8047 quantization.cpp:449] 1bit: 	0.1135
I0331 10:36:48.470614  8047 quantization.cpp:451] Fixed point FC weights: 
I0331 10:36:48.470633  8047 quantization.cpp:453] 16bit: 	0.9915
I0331 10:36:48.470654  8047 quantization.cpp:453] 8bit: 	0.9916
I0331 10:36:48.470674  8047 quantization.cpp:453] 4bit: 	0.9914
I0331 10:36:48.470695  8047 quantization.cpp:453] 2bit: 	0.9484
I0331 10:36:48.470716  8047 quantization.cpp:453] 1bit: 	0.1009
I0331 10:36:48.470736  8047 quantization.cpp:455] Fixed point layer outputs:
I0331 10:36:48.470751  8047 quantization.cpp:457] 16bit: 	0.9915
I0331 10:36:48.470770  8047 quantization.cpp:457] 8bit: 	0.9915
I0331 10:36:48.470790  8047 quantization.cpp:457] 4bit: 	0.9764
I0331 10:36:48.470809  8047 quantization.cpp:457] 2bit: 	0.1011
I0331 10:36:48.470829  8047 quantization.cpp:459] Fixed point net:
I0331 10:36:48.470844  8047 quantization.cpp:460] 2bit CONV weights,
I0331 10:36:48.470860  8047 quantization.cpp:461] 4bit FC weights,
I0331 10:36:48.470873  8047 quantization.cpp:462] 8bit layer outputs:
I0331 10:36:48.470887  8047 quantization.cpp:463] Accuracy: 0.9861
I0331 10:36:48.470902  8047 quantization.cpp:464] Please fine-tune.
```

##Parameters
* `model`: The network definition of the 32-bit floating point net.
* `weights`: The network parameters of the 32-bit floating point net.
* `trimming_mode`: The quantization strategy can be `fixed_point`, `mini_floating_point` or `power_of_2_weights`.
* `model_quantized`: The resulting quantized network definition.
* `error_margin`: The maximal accuracy drop compared to 32-bit floating point net.
* `gpu`: Ristretto supports both CPU and GPU mode.
* `iterations`: The number of batch iterations used for scoring the net.

##Trimming Modes

* **Dynamic Fixed Point**: First Ristretto analysis layer parameters and outputs. The tool chooses to use enough bits in the integer part to avoid saturation. Ristretto searches for the lowest possible bit-width for
    - parameters of convolutional layers
    - parameters of fully connected layers
    - layer outputs of convolutional and fully connected layers
* **Mini Floating Point**: First Ristretto analysis layer parameters and outputs. The tool chooses to use enough exponent bits to avoid saturation. Ristretto searches for the lowest possible bit-width for
    - parameters and outputs of convolutional and fully connected layers
* **Power-of-Two Parameters**: Ristretto benchmarks the network for 4-bit parameters. Ristretto chooses -8 and -1 for lowest and highest exponent, respectively.

