#!/usr/bin/env python
import os

trainval_jpeg_list = []
trainval_xml_list = []
test07_jpeg_list = []
test07_xml_list = []
test12_jpeg_list = []

for name in ["VOC2007", "VOC2012"]:
  voc_dir = os.path.join("VOCdevkit", name)
  txt_fold = os.path.join(voc_dir, "ImageSets/Main")
  jpeg_fold = os.path.join(voc_dir, "JPEGImages")
  xml_fold = os.path.join(voc_dir, "Annotations")
  for t in ["train.txt", "val.txt"]:
    file_path = os.path.join(txt_fold, t)
    with open(file_path, 'r') as fp:
      for line in fp:
        line = line.strip()
        trainval_jpeg_list.append(os.path.join(jpeg_fold, "{}.jpg".format(line)))
        trainval_xml_list.append(os.path.join(xml_fold, "{}.xml".format(line)))
        if not os.path.exists(trainval_jpeg_list[-1]):
          print trainval_jpeg_list[-1], "not exist"
        if not os.path.exists(trainval_xml_list[-1]):
          print trainval_xml_list[-1], "not exist"
  if name == "VOC2007":
    file_path = os.path.join(txt_fold, "test.txt")
    with open(file_path, 'r') as fp:
      for line in fp:
        line = line.strip()
        test07_jpeg_list.append(os.path.join(jpeg_fold, "{}.jpg".format(line)))
        test07_xml_list.append(os.path.join(xml_fold, "{}.xml".format(line)))
        if not os.path.exists(test07_jpeg_list[-1]):
          print test07_jpeg_list[-1], "not exist"
        if not os.path.exists(test07_xml_list[-1]):
          print test07_xml_list[-1], "not exist"
  elif name == "VOC2012":
    file_path = os.path.join(txt_fold, "test.txt")
    with open(file_path, 'r') as fp:
      for line in fp:
        line = line.strip()
        test12_jpeg_list.append(os.path.join(jpeg_fold, "{}.jpg".format(line)))
        if not os.path.exists(test12_jpeg_list[-1]):
          print test12_jpeg_list[-1], "not exist"

with open("trainval.txt", "w") as wr:
  for i in range(len(trainval_jpeg_list)):
    wr.write("{} {}\n".format(trainval_jpeg_list[i], trainval_xml_list[i]))

with open("test_2007.txt", "w") as wr:
  for i in range(len(test07_jpeg_list)):
    wr.write("{} {}\n".format(test07_jpeg_list[i], test07_xml_list[i]))

with open("test_2012.txt", "w") as wr:
  for i in range(len(test12_jpeg_list)):
    wr.write("{}\n".format(test12_jpeg_list[i]))

