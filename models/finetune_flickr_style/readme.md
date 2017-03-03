---
name: Finetuning CaffeNet on Flickr Style
caffemodel: finetune_flickr_style.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/finetune_flickr_style.caffemodel
license: non-commercial
sha1: b61b5cef7d771b53b0c488e78d35ccadc073e9cf
caffe_commit: 737ea5e936821b5c69f9c3952d72693ae5843370
gist_id: 034c6ac3865563b69e60
---

This model is trained exactly as described in `docs/finetune_flickr_style/readme.md`, using all 80000 images.
The final performance:

    I1017 07:36:17.370688 31333 solver.cpp:228] Iteration 100000, loss = 0.757952
    I1017 07:36:17.370730 31333 solver.cpp:247] Iteration 100000, Testing net (#0)
    I1017 07:36:34.248730 31333 solver.cpp:298]     Test net output #0: accuracy = 0.3916

This model was trained by Sergey Karayev @sergeyk

## License

The Flickr Style dataset contains only URLs to images.
Some of the images may have copyright.
Training a category-recognition model for research/non-commercial use may constitute fair use of this data, but the result should not be used for commercial purposes.
