# Caffe (CPM Data Layer)

## Files changed:
0. data_reader.hpp
0. data_transformer.hpp
0. data_reader.cpp
0. data_transformer.cpp

## Differences with new dataset
0. no need for nop (number other people)
0. initially no need for data manipulation (we already have a lot of data)
0. change the genJSON.py to include also 3D joint positions, as well as 2D joint positions
0. generate individual images from the train-set (subjects: 1,5,6,7,8)

## Fine-tuning a CNN for detection with Caffe
<pre>
  GLOG_logtostderr=1 build/tools/caffe train \
  -solver models/my_dir/caltech_finetune_solver_original.prototxt \
  -weights models/my_dir/bvlc_reference_caffenet.caffemodel \
  -gpu 0 2>&1 | tee results/log.txt
</pre>

## Multiple GPUs

Append after all the commands
<pre>
  --gpu=0,1
</pre>
for using two GPUs.
**NOTE**: each GPU runs the batchsize specified in your train_val.prototxt.  So if you go from 1 GPU to 2 GPU, your effective batchsize will double.  e.g. if your train_val.prototxt specified a batchsize of 256, if you run 2 GPUs your effective batch size is now 512.  So you need to adjust the batchsize when running multiple GPUs and/or adjust your solver params, specifically learning rate.

### Hardware Configuration Assumptions

The current implementation uses a tree reduction strategy.  e.g. if there are 4 GPUs in the system, 0:1, 2:3 will exchange gradients, then 0:2 (top of the tree) will exchange gradients, 0 will calculate
updated model, 0\-\>2, and then 0\-\>1, 2\-\>3. 

For best performance, P2P DMA access between devices is needed. Without P2P access, for example crossing PCIe root complex, data is copied through host and effective exchange bandwidth is greatly reduced.

Current implementation has a "soft" assumption that the devices being used are homogeneous.  In practice, any devices of the same general class should work together, but performance and total size is limited by the smallest device being used.  e.g. if you combine a TitanX and a GTX980, performance will be limited by the 980.  Mixing vastly different levels of boards, e.g. Kepler and Fermi, is not supported.

"nvidia-smi topo -m" will show you the connectivity matrix.  You can do P2P through PCIe bridges, but not across socket level links at this time, e.g. across CPU sockets on a multi-socket motherboard.

## CNN profiling

<pre>
  caffe time -model /path/to/file/structure.prototxt -iterations 10
</pre>
By default this is executed in CPU-mode. If instead a GPU-mode profiling is required, this is the command:
<pre>
  caffe time -model /path/to/file/structure.prototxt -gpu 0 -iterations 10
</pre>
