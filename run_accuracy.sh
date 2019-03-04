#Prepare data:
#Create raw lmdb of imagenet following https://github.com/intel/caffe/wiki/How-to-create-ImageNet-LMDB, please set "RESIZE=false" in  create_imagenet.sh to create raw lmdb
#ln -s ilsvrc12_val_lmdb "examples/imagenet/ilsvrc12_val_lmdb"

#expected result:
# loss3/top-1 = 0.75522
# loss3/top-5 = 0.926621

prototxt="./models/intel_optimized_models/resnet50_v1/resnet50_int8_acc.prototxt"
sed -i "1,/dim/s/dim.*/dim:$bs/" $prototxt

./build/tools/caffe test -model $prototxt -weights ./default_resnet_50_16_nodes_iter_56300.caffemodel --iterations 1000 --engine MKLDNN

