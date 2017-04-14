---
title: Model Zoo
---
# Caffe Model Zoo

Lots of researchers and engineers have made Caffe models for different tasks with all kinds of architectures and data: check out the [model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)!
These models are learned and applied for problems ranging from simple regression, to large-scale visual classification, to Siamese networks for image similarity, to speech and robotics applications.

To help share these models, we introduce the model zoo framework:

- A standard format for packaging Caffe model info.
- Tools to upload/download model info to/from Github Gists, and to download trained `.caffemodel` binaries.
- A central wiki page for sharing model info Gists.

## Where to get trained models

First of all, we bundle BAIR-trained models for unrestricted, out of the box use.
<br>
See the [BAIR model license](#bair-model-license) for details.
Each one of these can be downloaded by running `scripts/download_model_binary.py <dirname>` where `<dirname>` is specified below:

- **BAIR Reference CaffeNet** in `models/bvlc_reference_caffenet`: AlexNet trained on ILSVRC 2012, with a minor variation from the version as described in [ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Krizhevsky et al. in NIPS 2012. (Trained by Jeff Donahue @jeffdonahue)
- **BAIR AlexNet** in `models/bvlc_alexnet`: AlexNet trained on ILSVRC 2012, almost exactly as described in [ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Krizhevsky et al. in NIPS 2012. (Trained by Evan Shelhamer @shelhamer)
- **BAIR Reference R-CNN ILSVRC-2013** in `models/bvlc_reference_rcnn_ilsvrc13`: pure Caffe implementation of [R-CNN](https://github.com/rbgirshick/rcnn) as described by Girshick et al. in CVPR 2014. (Trained by Ross Girshick @rbgirshick)
- **BAIR GoogLeNet** in `models/bvlc_googlenet`: GoogLeNet trained on ILSVRC 2012, almost exactly as described in [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842) by Szegedy et al. in ILSVRC 2014. (Trained by Sergio Guadarrama @sguada)

**Community models** made by Caffe users are posted to a publicly editable [model zoo wiki page](https://github.com/BVLC/caffe/wiki/Model-Zoo).
These models are subject to conditions of their respective authors such as citation and license.
Thank you for sharing your models!

## Model info format

A caffe model is distributed as a directory containing:

- Solver/model prototxt(s)
- `readme.md` containing
    - YAML frontmatter
        - Caffe version used to train this model (tagged release or commit hash).
        - [optional] file URL and SHA1 of the trained `.caffemodel`.
        - [optional] github gist id.
    - Information about what data the model was trained on, modeling choices, etc.
    - License information.
- [optional] Other helpful scripts.

This simple format can be handled through bundled scripts or manually if need be.

### Hosting model info

Github Gist is a good format for model info distribution because it can contain multiple files, is versionable, and has in-browser syntax highlighting and markdown rendering.

`scripts/upload_model_to_gist.sh <dirname>` uploads non-binary files in the model directory as a Github Gist and prints the Gist ID. If `gist_id` is already part of the `<dirname>/readme.md` frontmatter, then updates existing Gist.

Try doing `scripts/upload_model_to_gist.sh models/bvlc_alexnet` to test the uploading (don't forget to delete the uploaded gist afterward).

Downloading model info is done just as easily with `scripts/download_model_from_gist.sh <gist_id> <dirname>`.

### Hosting trained models

It is up to the user where to host the `.caffemodel` file.
We host our BAIR-provided models on our own server.
Dropbox also works fine (tip: make sure that `?dl=1` is appended to the end of the URL).

`scripts/download_model_binary.py <dirname>` downloads the `.caffemodel` from the URL specified in the `<dirname>/readme.md` frontmatter and confirms SHA1.

## BAIR model license

The Caffe models bundled by the BAIR are released for unrestricted use.

These models are trained on data from the [ImageNet project](http://www.image-net.org/) and training data includes internet photos that may be subject to copyright.

Our present understanding as researchers is that there is no restriction placed on the open release of these learned model weights, since none of the original images are distributed in whole or in part.
To the extent that the interpretation arises that weights are derivative works of the original copyright holder and they assert such a copyright, UC Berkeley makes no representations as to what use is allowed other than to consider our present release in the spirit of fair use in the academic mission of the university to disseminate knowledge and tools as broadly as possible without restriction.
