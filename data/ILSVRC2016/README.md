### Preparation
#### ILSVRC2016
We encourage you to register [ILSVRC2016](http://image-net.org/challenges/LSVRC/2016) and download the DET dataset. By default, we assume the data is stored in `$HOME/data/ILSVRC` and will call it `$ILSVRC_ROOT`.

#### ILSVRC2015
If you choose to use ILSVRC2015 DET dataset, here are a few noticeable steps before running the following scripts:

1. There are a few problematic images. You can download the fixed ones [here](http://www.cs.unc.edu/~wliu/projects/SSD/ILSVRC2015_DET_fix.tar.gz).

2. You should download the [val1/val2 split](http://www.cs.unc.edu/~wliu/projects/SSD/ILSVRC2015_DET_val1_val2.tar.gz), courtesy of [Ross Girshick](http://people.eecs.berkeley.edu/~rbg), and put it in `$ILSVRC_ROOT/ImageSets/DET`.

### Remove an invalid file
Find the invalid image file `Data/DET/val/ILSVRC2013_val_00004542.JPEG`, and remove it.

### Create the LMDB file.
After you have downloaded the dataset, we can create the lmdb files.

  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval1.txt, val2.txt, val2_name_size.txt, test.txt and test_name_size.txt in data/ILSVRC2016/
  python data/ILSVRC2016/create_list.py
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval1, val2 and test with encoded original image:
  #   - $HOME/data/ILSVRC/lmdb/DET/ILSVRC2016_trainval1_lmdb
  #   - $HOME/data/ILSVRC/lmdb/DET/ILSVRC2016_val2_lmdb
  #   - $HOME/data/ILSVRC/lmdb/DET/ILSVRC2016_test_lmdb
  # and make soft links at examples/ILSVRC2016/
  ./data/ILSVRC2016/create_data.sh
  ```
