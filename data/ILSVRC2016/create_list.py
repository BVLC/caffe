import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys

HOMEDIR = os.path.expanduser("~")
CURDIR = os.path.dirname(os.path.realpath(__file__))

# If true, re-create all list files.
redo = False
# The root directory which holds all information of the dataset.
data_dir = "{}/data/ILSVRC".format(HOMEDIR)
# The directory name which holds the image sets.
imgset_dir = "ImageSets/DET"
# The direcotry which contains the images.
img_dir = "Data/DET"
img_ext = "JPEG"
# The directory which contains the annotations.
anno_dir = "Annotations/DET"
anno_ext = "xml"

train_list_file = "{}/trainval1.txt".format(CURDIR)
val_list_file = "{}/val2.txt".format(CURDIR)
val_name_size_file = "{}/val2_name_size.txt".format(CURDIR)
test_list_file = "{}/test.txt".format(CURDIR)
test_name_size_file = "{}/test_name_size.txt".format(CURDIR)

# Create training set.
# We follow Ross Girschick's split in R-CNN.
if redo or not os.path.exists(train_list_file):
    datasets = ["train", "val1"]
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n").split(" ")[0]
                subset = name.split("/")[0].split("_")[1]
                anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                # Ignore image if it does not have annotation. These are the negative images in ILSVRC.
                if not os.path.exists("{}/{}".format(data_dir, anno_file)):
                    continue
                img_file = "{}/{}/{}.{}".format(img_dir, subset, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file))
                img_files.append(img_file)
                anno_files.append(anno_file)
    # Shuffle the images.
    idx = [i for i in xrange(len(img_files))]
    shuffle(idx)
    with open(train_list_file, "w") as f:
        for i in idx:
            f.write("{} {}\n".format(img_files[i], anno_files[i]))

if redo or not os.path.exists(val_list_file):
    datasets = ["val2"]
    subset = "val"
    img_files = []
    anno_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n").split(" ")[0]
                img_file = "{}/{}/{}.{}".format(img_dir, subset, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file))
                anno_file = "{}/{}/{}.{}".format(anno_dir, subset, name, anno_ext)
                assert os.path.exists("{}/{}".format(data_dir, anno_file))
                img_files.append(img_file)
                anno_files.append(anno_file)
    with open(val_list_file, "w") as f:
        for i in xrange(len(img_files)):
            f.write("{} {}\n".format(img_files[i], anno_files[i]))

if redo or not os.path.exists(val_name_size_file):
    dataset = 'val2'
    imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
    cmd = "{}/../../build/tools/get_image_size --name_id_file={} {} {} {}".format(
            CURDIR, imgset_file, data_dir, val_list_file, val_name_size_file)
    print cmd
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

if redo or not os.path.exists(test_list_file):
    datasets = ["test"]
    subset = "test"
    img_files = []
    for dataset in datasets:
        imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n").split(" ")[0]
                img_file = "{}/{}/{}.{}".format(img_dir, subset, name, img_ext)
                assert os.path.exists("{}/{}".format(data_dir, img_file))
                img_files.append(img_file)
    with open(test_list_file, "w") as f:
        for i in xrange(len(img_files)):
            f.write("{} 0\n".format(img_files[i]))

if redo or not os.path.exists(test_name_size_file):
    dataset = 'test'
    imgset_file = "{}/{}/{}.txt".format(data_dir, imgset_dir, dataset)
    cmd = "{}/../../build/tools/get_image_size --name_id_file={} {} {} {}".format(
            CURDIR, imgset_file, data_dir, test_list_file, test_name_size_file)
    print cmd
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
