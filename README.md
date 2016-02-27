# OpenCL Caffe

**This is an experimental, community-maintained branch led by Fabian Tschopp (@naibaf7). It is a work-in-progress.**

This branch of Caffe contains an OpenCL backend and additional layers for fast image segmentation.
This work is partially supported by:
- AMD
- HHMI Janelia
- UZH, INI
- ETH Zurich

For a C++ frontend and models to use for image segmentation with this fork, see:
- Frontend: https://github.com/naibaf7/caffe_neural_tool
- Models: https://github.com/naibaf7/caffe_neural_models

## OpenCL Backend

The backend is supposed to work with all vendors. Note however there may be problems with libOpenCL.so provided by nVidia.
It is therefore recommended to install another OpenCL implementation after installing nVidia drivers. Possibilities are:
- Intel OpenCL, recommended if you have an Intel CPU along the nVidia GPU.
- AMD APP SDK (OpenCL), recommended if you have an AMD GPU or CPU.

## Technical Report
Available on arXiv:
http://arxiv.org/abs/1509.03371

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.
