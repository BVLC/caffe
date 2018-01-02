---
title: Multi-GPU Usage, Hardware Configuration Assumptions, and Performance
---

# Multi-GPU Usage

Currently Multi-GPU is only supported via the C/C++ paths and only for training.

The GPUs to be used for training can be set with the "-gpu" flag on the command line to the 'caffe' tool.  e.g. "build/tools/caffe train --solver=models/bvlc_alexnet/solver.prototxt --gpu=0,1" will train on GPUs 0 and 1.

**NOTE**: each GPU runs the batchsize specified in your train_val.prototxt.  So if you go from 1 GPU to 2 GPU, your effective batchsize will double.  e.g. if your train_val.prototxt specified a batchsize of 256, if you run 2 GPUs your effective batch size is now 512.  So you need to adjust the batchsize when running multiple GPUs and/or adjust your solver params, specifically learning rate.

# Hardware Configuration Assumptions

The current implementation uses a tree reduction strategy.  e.g. if there are 4 GPUs in the system, 0:1, 2:3 will exchange gradients, then 0:2 (top of the tree) will exchange gradients, 0 will calculate
updated model, 0\-\>2, and then 0\-\>1, 2\-\>3.

For best performance, P2P DMA access between devices is needed. Without P2P access, for example crossing PCIe root complex, data is copied through host and effective exchange bandwidth is greatly reduced.

Current implementation has a "soft" assumption that the devices being used are homogeneous.  In practice, any devices of the same general class should work together, but performance and total size is limited by the smallest device being used.  e.g. if you combine a TitanX and a GTX980, performance will be limited by the 980.  Mixing vastly different levels of boards, e.g. Kepler and Fermi, is not supported.

"nvidia-smi topo -m" will show you the connectivity matrix.  You can do P2P through PCIe bridges, but not across socket level links at this time, e.g. across CPU sockets on a multi-socket motherboard.

# Scaling Performance

Performance is **heavily** dependent on the PCIe topology of the system, the configuration of the neural network you are training, and the speed of each of the layers.  Systems like the DIGITS DevBox have an optimized PCIe topology (X99-E WS chipset).  In general, scaling on 2 GPUs tends to be ~1.8X on average for networks like AlexNet, CaffeNet, VGG, GoogleNet.  4 GPUs begins to have falloff in scaling.  Generally with "weak scaling" where the batchsize increases with the number of GPUs you will see 3.5x scaling or so.  With "strong scaling", the system can become communication bound, especially with layer performance optimizations like those in [cuDNNv3](http://nvidia.com/cudnn), and you will likely see closer to mid 2.x scaling in performance.  Networks that have heavy computation compared to the number of parameters tend to have the best scaling performance.
