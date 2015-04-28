#!/bin/sh 

IMG_DIR=/rmt/data/pascal/VOCdevkit/VOC2012/PPMImages
FEATURE_DIR=/rmt/work/deeplabel/exper/voc12/features/vgg128_noup/val/fc8/bin
SAVE_DIR=/rmt/work/deeplabel/exper/voc12/features/vgg128_noup/val/fc8/post_densecrf

MAX_ITER=10

POS_X_STD=3
POS_Y_STD=3
POS_W=3

Bi_X_STD=60
Bi_Y_STD=60
Bi_R_STD=20
Bi_G_STD=20
Bi_B_STD=20
Bi_W=10


./prog_refine_pascal -id $(IMG_DIR) -fd $(FEATURE_DIR) -sd $(SAVE_DIR) -i $(MAX_ITER) -px $(POS_X_STD) -py $(POS_Y_STD) -pw $(POS_W) -bx $(Bi_X_STD) -by $(Bi_Y_STD) -br $(Bi_R_STD) -bg $(Bi_G_STD) -bb $(Bi_B_STD) -bw $(Bi_W)

