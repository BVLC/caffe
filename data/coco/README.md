### Preparation
1. Download Images and Annotations from [MSCOCO](http://mscoco.org/dataset/#download). By default, we assume the data is stored in `$DATAPATH/data/coco`
make sure extracted image directories (train2014, val2014 test2015) are inside
'/images/' directory next to Annotations directory.
And download also:
http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/instances_minival2014.json.zip
http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/instances_valminusminival2014.json.zip 
unzip instances_minival2014.json.zip
unzip instances_valminusminival2014.json.zip
mv instances_minival2014.json instances_valminusminival2014.json ./coco/annotations/
2. Get the coco code. We will call the directory that you cloned coco into `$COCO_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/coco.git
  cd coco
  git checkout dev
  git apply $CAFFE_ROOT/data/coco/diff.patch
  ```

3. Build the coco code.
  ```Shell
  cd PythonAPI
  python setup.py build_ext --inplace
  ```

4. Split the annotation to many files per image and get the image size info.
  ```Shell
  # Check scripts/batch_split_annotation.py and change settings accordingly.
  python scripts/batch_split_annotation.py
  # Create the minival2014_name_size.txt and test-dev2015_name_size.txt in $CAFFE_ROOT/data/coco
  python scripts/batch_get_image_size.py
  ```

5. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the minival.txt, testdev.txt, test.txt, train.txt in data/coco/
  python data/coco/create_list.py
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for minival, testdev, test, and train with encoded original image:
  #   - $DATAPATH/data/coco/lmdb/coco_minival_lmdb
  #   - $DATAPATH/data/coco/lmdb/coco_testdev_lmdb
  #   - $DATAPATH/data/coco/lmdb/coco_test_lmdb
  #   - $DATAPATH/data/coco/lmdb/coco_train_lmdb
  # and make soft links at examples/coco/
  ./data/coco/create_data.sh
  ```
