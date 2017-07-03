please download https://pjreddie.com/media/files/yolo-voc.weights  to yolo416/yolo.weights

1. how to change yolo weight to caffemodel
  python convert_yolo_to_caffemodel.py
2.how to fused
  python ${CAFFE_ROOT}/tools/inference-optimize/model_fuse.py --indefinition yolo416/yolo_deploy.prototxt --outdefinition yolo416/fuse_yolo_deploy.prototxt --inmodel yolo416/yolo.caffemodel --outmodel yolo416/fuse_yolo.caffemodel

Then you can use the fuse_yolo_deploy.prototxt and fuse_yolo.caffemodel to do inference which is much faster (1.5x) than the non-fused version.

!!! all python must run in models/yolo folder
