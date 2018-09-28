# CAFFE for YOLO9000

## Reference

> YOLO9000: Better, Faster, Stronger

> http://pjreddie.com/yolo9000/

> https://github.com/yeahkun/caffe-yolo

> https://github.com/weiliu89/caffe/tree/ssd

> https://github.com/choasup/caffe-yolo9000
## Usage

### Caffe 
```Shell
   You can compile Caffe with CMake or make method.
```

### Data preparation
Like SSD data setting.
```Shell
  cd data/VOC0712  
  vim create_data.sh
  ./create_data.sh 
```

### Train
```Shell
  cd models/intel_optimized_models/yolo
  # download pretrain_model
```  
  > https://pan.baidu.com/s/1c71EB-6A1xQb2ImOISZiHA password: 9u5v
```
  # change pretrain model related path in script train.sh
  cd examples/yolo/
  vim train_yolo_V2.sh
  
  ./train_yolo_V2.sh
```

### Eval mAP
```Shell
   cd examples/yolo/

   ./test_yolo_V2.sh
```
