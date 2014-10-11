#!/usr/bin/env sh
#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=../../../build/tools
MODEL=../pose_caffenet_train_iter_200000
PROTOTXT=../caffenet-pose-lsp-train-val.prototxt
# CONV 1
#LAYER=conv1
#LEVELDB=features_conv1_0919

# FC8-POSE
#LAYER=fc8_pose
#LEVELDB=features_fc8_0919

# LABEL
LAYER=groundtruth
LEVELDB=../lspc_test_leveldb_old

BATCHSIZE=128

# args for LEVELDB to MAT
#DIM=290400 # conv1
DIM=42 # fc8-pose
OUT=$LAYER/features.mat
BATCHNUM=1

mkdir $LAYER
#$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE GPU
python ../../../build/wyang/leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
