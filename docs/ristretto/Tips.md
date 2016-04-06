#Helpful tips

* **Nets with multiple accuracy layers**: The Ristretto tools condenses a neural network, according to the user-defined error margin. The tool assumes that the accuracy in question is the very first accuracy score in the network description file. If you wish to trim a network according to a different accuracy score, please adjust `score_number` default value in include/ristretto/quantization.hpp::RunForwardBatches(...).

#Limitations

* **Tests**: All Caffe-layers have unit tests. Ristretto doesn't have test cases yet.
