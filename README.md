# Ristretto

Ristretto is an automated framework for the **quantization and compression** of Convolutional Neural Networks (CNNs).

The primary goal of this project is to enable **fast and energy-efficient inference** of artificial intelligence models on resource-constrained hardware platforms, such as FPGAs and custom accelerators. By reducing the bit-width of network parameters and activations, Ristretto significantly lowers the memory footprint and computational requirements of deep learning models with minimal impact on accuracy.

## Key Features

* **Hardware-Oriented Approximation:** Automatically simulates the effects of reduced precision on CNN performance.
* **Mixed-Precision Quantization:** Supports different bit-widths for different layers to optimize the trade-off between speed and accuracy.
* **Dynamic Fixed-Point:** Uses a flexible fixed-point format that adapts to the dynamic range of different network components.
* **Automated Fine-Tuning:** Provides tools to retrain quantized networks to recover accuracy lost during the approximation process.

## Framework Origin

Ristretto was developed by [Philipp Gysel](https://www.linkedin.com/in/pgysel/) during his Master's thesis at UC Davis. The open-source project was introduced alongside a workshop paper at ICLR titled [Hardware-oriented Approximation of Convolutional Neural Networks](https://arxiv.org/pdf/1604.03168).

Ristretto is a fork of the [Caffe deep learning framework](https://caffe.berkeleyvision.org/). It extends the original Caffe implementation by introducing specialized layers and scoring functions designed for network approximation and hardware-friendly deployment.

## Cite this Research

If you find this open-source project useful for your own research, please consider citing this paper:

```bibtex
@article{gysel2018ristretto,
  title={Ristretto: A Framework for Empirical Study of Resource-Efficient Inference in Convolutional Neural Networks},
  author={Gysel, Philipp and Pimentel, Jon and Motamedi, Mohammad and Ghiasi, Soheil},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={29},
  number={11},
  pages={5784--5789},
  year={2018},
  publisher={IEEE},
  doi={10.1109/TNNLS.2018.2808319}
}
```

## Other Resources

* **Ristretto webpage**: Please check out the [official Ristretto webpage](https://leps.sf.ucdavis.edu/ristretto-cnn-approximation) for more information. The official webpage contains usage examples and comprehensive walkthrough guides.

* **Ristretto forum**: If you have any technical questions about Ristretto, please use the [Google Group](https://groups.google.com/g/ristretto-users).

## Notes on Caffe

As mentioned, Ristretto is based on [Caffe](https://caffe.berkeleyvision.org/), a deep learning framework created by Berkeley AI Research ([BAIR](https://bair.berkeley.edu/)) and community contributors. Its original author is [Yangqing Jia](https://daggerfs.com/). Below you can find the licensing and citation instructions from Caffe.

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

Please cite Caffe in your publications if it helps your research:

```bibtex
@article{jia2014caffe,
  Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
  Journal = {arXiv preprint arXiv:1408.5093},
  Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
  Year = {2014}
}
```