---
title: Model Zoo
---
# Caffe Model Zoo

Lots of people have used Caffe to train models of different architectures and applied to different problems, ranging from simple regression to AlexNet-alikes to Siamese networks for image similarity to speech applications.
To lower the friction of sharing these models, we introduce the model zoo framework:

- A standard format for packaging Caffe model info.
- Tools to upload/download model info to/from Github Gists, and to download trained `.caffemodel` binaries.
- A central wiki page for sharing model info Gists.

## Where to get trained models

First of all, we provide some trained models out of the box.
Each one of these can be downloaded by running `scripts/download_model_binary.py <dirname>` where `<dirname>` is specified below:

- **BVLC Reference CaffeNet** in `models/bvlc_reference_caffenet`: AlexNet trained on ILSVRC 2012, with a minor variation from the version as described in the NIPS 2012 paper.
- **BVLC AlexNet** in `models/bvlc_alexnet`: AlexNet trained on ILSVRC 2012, almost exactly as described in NIPS 2012.
- **BVLC Reference R-CNN ILSVRC-2013** in `models/bvlc_reference_rcnn_ilsvrc13`: pure Caffe implementation of [R-CNN](https://github.com/rbgirshick/rcnn).

User-provided models are posted to a public-editable [wiki page](https://github.com/BVLC/caffe/wiki/Model-Zoo).

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

## Hosting model info

Github Gist is a good format for model info distribution because it can contain multiple files, is versionable, and has in-browser syntax highlighting and markdown rendering.

- `scripts/upload_model_to_gist.sh <dirname>`: uploads non-binary files in the model directory as a Github Gist and prints the Gist ID. If `gist_id` is already part of the `<dirname>/readme.md` frontmatter, then updates existing Gist.

Try doing `scripts/upload_model_to_gist.sh models/bvlc_alexnet` to test the uploading (don't forget to delete the uploaded gist afterward).

Downloading model info is done just as easily with `scripts/download_model_from_gist.sh <gist_id> <dirname>`.

### Hosting trained models

It is up to the user where to host the `.caffemodel` file.
We host our BVLC-provided models on our own server.
Dropbox also works fine (tip: make sure that `?dl=1` is appended to the end of the URL).

- `scripts/download_model_binary.py <dirname>`: downloads the `.caffemodel` from the URL specified in the `<dirname>/readme.md` frontmatter and confirms SHA1.
