---
name: BAIR/BVLC Reference RCNN ILSVRC13 Model
caffemodel: bvlc_reference_rcnn_ilsvrc13.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/bvlc_reference_rcnn_ilsvrc13.caffemodel
license: unrestricted
sha1: bdd8abb885819cba5e2fe1eb36235f2319477e64
caffe_commit: a7e397abbda52c0b90323c23ab95bdeabee90a98
---

The pure Caffe instantiation of the [R-CNN](https://github.com/rbgirshick/rcnn) model for ILSVRC13 detection.
This model was made by transplanting the R-CNN SVM classifiers into a `fc-rcnn` classification layer, provided here as an off-the-shelf Caffe detector.
Try the [detection example](http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/detection.ipynb) to see it in action.

*N.B. For research purposes, make use of the official R-CNN package and not this example.*

This model was trained by Ross Girshick @rbgirshick

## License

This model is released for unrestricted use.
