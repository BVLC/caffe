---
name: BVLC AlexNet Model
caffemodel: bvlc_alexnet.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
license: non-commercial
sha1: 9116a64c0fbe4459d18f4bb6b56d647b63920377
caffe_commit: 709dc15af4a06bebda027c1eb2b3f3e3375d5077
---

This model is a replication of the model described in the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) publication.

Differences:
- not training with the relighting data-augmentation;
- initializing non-zero biases to 0.1 instead of 1 (found necessary for training, as initialization to 1 gave flat loss).

The bundled model is the iteration 360,000 snapshot.
The best validation performance during training was iteration 358,000 with validation accuracy 57.258% and loss 1.83948.
This model obtains a top-1 accuracy 57.1% and a top-5 accuracy 80.2% on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror, should obtain a bit higher accuracy.)

## License

The data used to train this model comes from the ImageNet project, which distributes its database to researchers who agree to a following term of access:
"Researcher shall use the Database only for non-commercial research and educational purposes."
Accordingly, this model is distributed under a non-commercial license.
