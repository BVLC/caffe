---
name: BAIR/BVLC CaffeNet Model
caffemodel: bvlc_reference_caffenet.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
license: unrestricted
sha1: 4c8d77deb20ea792f84eb5e6d0a11ca0a8660a46
caffe_commit: 709dc15af4a06bebda027c1eb2b3f3e3375d5077
---

This model is the result of following the Caffe [ImageNet model training instructions](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html).
It is a replication of the model described in the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) publication with some differences:

- not training with the relighting data-augmentation;
- the order of pooling and normalization layers is switched (in CaffeNet, pooling is done before normalization).

This model is snapshot of iteration 310,000.
The best validation performance during training was iteration 313,000 with validation accuracy 57.412% and loss 1.82328.
This model obtains a top-1 accuracy 57.4% and a top-5 accuracy 80.4% on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror, should obtain a bit higher accuracy still.)

This model was trained by Jeff Donahue @jeffdonahue

## License

This model is released for unrestricted use.
