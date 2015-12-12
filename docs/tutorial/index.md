---
title: Caffe Tutorial
---
# Caffe Tutorial

Caffe is a deep learning framework and this tutorial explains its philosophy, architecture, and usage.
This is a practical guide and framework introduction, so the full frontier, context, and history of deep learning cannot be covered here.
While explanations will be given where possible, a background in machine learning and neural networks is helpful.

## Philosophy

In one sip, Caffe is brewed for

- Expression: models and optimizations are defined as plaintext schemas instead of code.
- Speed: for research and industry alike speed is crucial for state-of-the-art models and massive data.
- Modularity: new tasks and settings require flexibility and extension.
- Openness: scientific and applied progress call for common code, reference models, and reproducibility.
- Community: academic research, startup prototypes, and industrial applications all share strength by joint discussion and development in a BSD-2 project.

and these principles direct the project.

## Tour

- [Nets, Layers, and Blobs](net_layer_blob.html): the anatomy of a Caffe model.
- [Forward / Backward](forward_backward.html): the essential computations of layered compositional models.
- [Loss](loss.html): the task to be learned is defined by the loss.
- [Solver](solver.html): the solver coordinates model optimization.
- [Layer Catalogue](layers.html): the layer is the fundamental unit of modeling and computation -- Caffe's catalogue includes layers for state-of-the-art models.
- [Interfaces](interfaces.html): command line, Python, and MATLAB Caffe.
- [Data](data.html): how to caffeinate data for model input.

For a closer look at a few details:

- [Caffeinated Convolution](convolution.html): how Caffe computes convolutions.

## Deeper Learning

There are helpful references freely online for deep learning that complement our hands-on tutorial.
These cover introductory and advanced material, background and history, and the latest advances.

The [Tutorial on Deep Learning for Vision](https://sites.google.com/site/deeplearningcvpr2014/) from CVPR '14 is a good companion tutorial for researchers.
Once you have the framework and practice foundations from the Caffe tutorial, explore the fundamental ideas and advanced research directions in the CVPR '14 tutorial.

A broad introduction is given in the free online draft of [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) by Michael Nielsen. In particular the chapters on using neural nets and how backpropagation works are helpful if you are new to the subject.

These recent academic tutorials cover deep learning for researchers in machine learning and vision:

- [Deep Learning Tutorial](http://www.cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf) by Yann LeCun (NYU, Facebook) and Marc'Aurelio Ranzato (Facebook). ICML 2013 tutorial.
- [LISA Deep Learning Tutorial](http://deeplearning.net/tutorial/deeplearning.pdf) by the LISA Lab directed by Yoshua Bengio (U. Montréal).

For an exposition of neural networks in circuits and code, check out [Understanding Neural Networks from a Programmer's Perspective](http://karpathy.github.io/neuralnets/) by Andrej Karpathy (Stanford).
