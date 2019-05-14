#Prepare data:
#Create raw lmdb of imagenet following https://github.com/intel/caffe/wiki/How-to-create-ImageNet-LMDB, please set "RESIZE=false" in  create_imagenet.sh to create raw lmdb
#ln -s ilsvrc12_val_lmdb "examples/imagenet/ilsvrc12_val_lmdb"

#expected result:
# loss3/top-1 = 0.75522
# loss3/top-5 = 0.926621

prototxt="./models/intel_optimized_models/int8/resnet50_int8_acc_clx_winograd.prototxt"

./build/tools/caffe test -model $prototxt -weights ./resnet50_clx_force_u8.caffemodel --iterations 1000 --engine MKLDNN

