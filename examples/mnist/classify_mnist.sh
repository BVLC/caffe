#!/usr/bin/env sh

../../python/classify.py --model_def ./lenet.prototxt \
                         --pretrained_model ./lenet_iter_10000 \
                         --mean_file='' \
                         --center_only --images_dim 28,28 --gpu --channel_swap '0' ./mnist-predict-100-twos.npy predictions

