1.how to fused
  python yolo_fuse.py --indefinition yolo416/yolo_deploy.prototxt --outdefinition yolo416/fuse_yolo_deploy.prototxt --inmodel yolo416/yolo.caffemodel --outmodel yolo416/fuse_yolo.caffemodel
2.how to change yolo weight to caffemodel
  python convert_yolo_to_caffemodel.py

!!! all python must run in models/yolo folder
please download https://pjreddie.com/media/files/yolo-voc.weights  to yolo416/yolo.weights 