# hash-caffe

This is a caffe repository for learning to hash. We fork the repository from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Add `multi label layer` which enable ImageDataLayer to process multi-label dataset.
- Add `pairwise loss layer` and `quantization loss layer` described in paper "Deep Hashing Network for Efficient Similarity Retrieval".

Data Preparation
---------------
In `data/nus_wide/train.txt`, we give an example to show how to prepare training data. In `data/nus_wide/parallel/`, the list of testing and database images are splitted to 12 parts, which could be processed parallelly when predicting.

Training Model
---------------

In `models/DHN/nus_wide/`, we give an example to show how to train hash model. In this model, we use pairwise loss and quantization loss as loss functions.

The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model. If the NUS\_WIDE dataset and pre-trained caffemodel is prepared, the example can be run with the following command:
```
"./build/tools/caffe train -solver models/DHN/nus_wide/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
```

Parameter Tuning
---------------
In pairwise loss layer and quantization loss layer, parameter `loss_weight` can be tuned to give them different weights.

Predicting
---------------
In `models/DHN/predict/predict_parallel.py`, we give an example to show how to evaluate the trained hash model.

Citation
---------------
    @inproceedings{DBLP:conf/aaai/ZhuL0C16,
      author    = {Han Zhu and
                   Mingsheng Long and
                   Jianmin Wang and
                   Yue Cao},
      title     = {Deep Hashing Network for Efficient Similarity Retrieval},
      booktitle = {Proceedings of the Thirtieth {AAAI} Conference on Artificial Intelligence,
                   February 12-17, 2016, Phoenix, Arizona, {USA.}},
      pages     = {2415--2421},
      year      = {2016},
      crossref  = {DBLP:conf/aaai/2016},
      url       = {http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12039},
      timestamp = {Thu, 21 Apr 2016 19:28:00 +0200},
      biburl    = {http://dblp.uni-trier.de/rec/bib/conf/aaai/ZhuL0C16},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
