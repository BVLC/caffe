---
name: BVLC GoogleNet Model
caffemodel: bvlc_googlenet.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
license: unrestricted
sha1: 405fc5acd08a3bb12de8ee5e23a96bec22f08204
caffe_commit: bc614d1bd91896e3faceaf40b23b72dab47d44f5
---

This model is a replication of the model described in the [GoogleNet](http://arxiv.org/abs/1409.4842) publication. We would like to thank Christian Szegedy for all his help in the replication of GoogleNet model.

Differences:
- not training with the relighting data-augmentation;
- not training with the scale or aspect-ratio data-augmentation;
- uses "xavier" to initialize the weights instead of "gaussian";
- quick_solver.prototxt uses a different learning rate decay policy than the original solver.prototxt, that allows a much faster training (60 epochs vs 250 epochs);

The bundled model is the iteration 2,400,000 snapshot (60 epochs) using quick_solver.prototxt

This bundled model obtains a top-1 accuracy 68.7% (31.3% error) and a top-5 accuracy 88.9% (11.1% error) on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror, should obtain a bit higher accuracy.)

Timings for bvlc_googlenet with cuDNN using batch_size:128 on a K40c:
 - Average Forward pass: 562.841 ms.
 - Average Backward pass: 1123.84 ms.
 - Average Forward-Backward: 1688.8 ms.

This model was trained by Sergio Guadarrama @sguada

## License

This model is released for unrestricted use.
