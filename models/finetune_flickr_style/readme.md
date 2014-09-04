---
name: Finetuning CaffeNet on Flickr Style
caffemodel: finetune_flickr_style.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/finetune_flickr_style.caffemodel
license: non-commercial
sha1: 443ad95a61fb0b5cd3cee55951bcc1f299186b5e
caffe_commit: 41751046f18499b84dbaf529f64c0e664e2a09fe
gist_id: 034c6ac3865563b69e60
---

This model is trained exactly as described in `docs/finetune_flickr_style/readme.md`, using all 80000 images.
The final performance on the test set:

    I0903 18:40:59.211707 11585 caffe.cpp:167] Loss: 0.407405
    I0903 18:40:59.211717 11585 caffe.cpp:179] accuracy = 0.9164

## License

The Flickr Style dataset contains only URLs to images.
Some of the images may have copyright.
Training a category-recognition model for research/non-commercial use may constitute fair use of this data, but the result should not be used for commercial purposes.
