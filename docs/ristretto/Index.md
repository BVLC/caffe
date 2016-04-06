#Ristretto
Ristretto is an automated CNN-approximation tool which condenses 32-bit floating point networks. Ristretto is an extention of [Caffe](http://caffe.berkeleyvision.org/) and allows to test, train and finetune networks with limited numerical precision.

##Ristretto In a Minute
* **Ristretto Tool**: The Ristretto tool performs automatic network quantization and scoring, using different bit-widths for number representation, to find a good balance between compression rate and network accuracy.
* **Ristretto Layers**: Ristretto reimplements Caffe-layers with quantized numbers.
* **Testing and Training**: Thanks to Ristretto's smooth integration into Caffe, network description files can be manually changed to quantize different layers. The bit-width used for different layers as well as other parameters can be set in the network's prototxt file. This allows to directly test and train condensed networks, without any need of recompilation.

##Approximation Schemes
Ristretto allows for three different quantization strategies to approximate Convolutional Neural Networks:

* **Dynamic Fixed Point**: A modified fixed-point format with more flexibility.
* **Mini Floating Point**: Bit-width reduced floating point numbers.
* **Power-of-two parameters**: Layers with power-of-two parameters don't need any multipliers, when implemented in hardware.

##Documentation
* [SqueezeNet Example](SqueezeNet.md): Quantization, fine-tuning and benchmarking of SqueezeNet.
* [Ristretto Layers, Benchmarking and Finetuning](Layers_Test_Train.md): Implementation details of Ristretto.
* [Approximation Schemes](Approximations.md)
* [Ristretto Layer Catalogue](Layer_Catalogue.md): List of layers that can be approximated by Ristretto.
* [Ristretto Tool](Ristretto_Tool.md): The command line tool and its parameters.
* [Tips and Tricks](Tips.md)

##Cite us
Our approximation framework is presented in [this](http://beta.openreview.net/forum?id=81DnLL9OEI6O2Pl0UV1w) ICLR'16 workshop paper.
