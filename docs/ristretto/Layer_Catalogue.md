#Ristretto Layer Catalogue

Ristretto supports the approximation of different layer types of convolutional neural networks. The next two tables explain how different layers can be quantized, and how this quantization affects different parts of a layer.

##Quantization Support by Layer

| Layer Type | Fixed Point | Floating Point | Power-of-two parameters |
|:---------|:-----------------:|:----------------:|:-------------:|
| Convolution | ![](Checkmark.png) | ![](Checkmark.png) | ![](Checkmark.png) |
| Fully Connected | ![](Checkmark.png) | ![](Checkmark.png) | ![](Checkmark.png) |
| LRN | | ![](Checkmark.png) |  |

Local Response Normalization (LRN) layers only support quantization to mini floating point. This layer type uses "strict arithmetic", i.e. all intermediate results are quantized.

##Quantization of Parameters and Layer Outputs

| Quantization | Parameters | Layer Outputs |
|:------|:------:|:------:|
| Fixed Point | ![](Checkmark.png) | ![](Checkmark.png) |
| Floating Point | ![](Checkmark.png) | ![](Checkmark.png) |
| Power-of-two parameters | ![](Checkmark.png) | |

##Google Protocol Buffer Fields

Just as with Caffe, you need to define Ristretto models using protocol buffer definition files (*.prototxt). All Ristretto layer parameters are defined in caffe.proto.

###Common fields
* `type`: Ristretto supports the following layers: `ConvolutionRistretto`, `FcRistretto` (fully connected layer) and `LRNRistretto`.
* Parameters:
	- `precision` [default FIXED_POINT]: the quantization strategy should be FIXED_POINT, MINI_FLOATING_POINT or POWER_2_WEIGHTS
	- `rounding_scheme` [default NEAREST]: the rounding scheme used for quantization should be either round-nearest-even (NEAREST) or round-stochastic (STOCHASTIC)

###Dynamic Fixed Point
* Precision type: `FIXED_POINT`
* Parameters:
	- `bw_layer_out` [default 32]: the number of bits used for representing layer outputs
	- `bw_params` [default 32]: the number of bits used for representing layer parameters
	- `fl_layer_out` [default 16]: fractional bits used for representing layer outputs
	- `fl_params` [default 16]: fractional bits used for representing layer parameters
* The default values correspond to 32-bit fixed point numbers with 16 integer and fractional bits.

###Mini Floating Point
* Precision type: `MINI_FLOATING_POINT`
* Parameters:
	- `mant_bits` [default: 23]: the number of bits used for representing the mantissa
	- `exp_bits` [default: 8]: the number of bits used for representing the exponent
* The default values correspond to single precision format

###Power-of-Two Parameters
* Precision type: `POWER_2_WEIGHTS`
* Parameters:
	- `exp_min` [default: -8] : The minimum exponent used
	- `exp_max` [default: -1] : The maximum exponent used
* The default values result in a format that can be represented with 4 bits in hardware (1 sign bit and 3 bits for exponent value)

##Example Ristretto Layer
```
layer {
  name: "norm1"
  type: "LRNRistretto"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
  quantization_param {
    precision: MINI_FLOATING_POINT
    mant_bits: 10
    exp_bits: 7
  }
}
```

