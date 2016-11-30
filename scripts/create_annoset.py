import argparse
import os
import shutil
import subprocess
import sys

from caffe.proto import caffe_pb2
from google.protobuf import text_format

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Create AnnotatedDatum database")
  parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
  parser.add_argument("listfile",
      help="The file which contains image paths and annotation info.")
  parser.add_argument("outdir",
      help="The output directory which stores the database file.")
  parser.add_argument("exampledir",
      help="The directory to store the link of the database files.")
  parser.add_argument("--redo", default = False, action = "store_true",
      help="Recreate the database.")
  parser.add_argument("--anno-type", default = "classification",
      help="The type of annotation {classification, detection}.")
  parser.add_argument("--label-type", default = "xml",
      help="The type of label file format for detection {xml, json, txt}.")
  parser.add_argument("--backend", default = "lmdb",
      help="The backend {lmdb, leveldb} for storing the result")
  parser.add_argument("--check-size", default = False, action = "store_true",
      help="Check that all the datum have the same size.")
  parser.add_argument("--encode-type", default = "",
      help="What type should we encode the image as ('png','jpg',...).")
  parser.add_argument("--encoded", default = False, action = "store_true",
      help="The encoded image will be save in datum.")
  parser.add_argument("--gray", default = False, action = "store_true",
      help="Treat images as grayscale ones.")
  parser.add_argument("--label-map-file", default = "",
      help="A file with LabelMap protobuf message.")
  parser.add_argument("--min-dim", default = 0, type = int,
      help="Minimum dimension images are resized to.")
  parser.add_argument("--max-dim", default = 0, type = int,
      help="Maximum dimension images are resized to.")
  parser.add_argument("--resize-height", default = 0, type = int,
      help="Height images are resized to.")
  parser.add_argument("--resize-width", default = 0, type = int,
      help="Width images are resized to.")
  parser.add_argument("--shuffle", default = False, action = "store_true",
      help="Randomly shuffle the order of images and their labels.")
  parser.add_argument("--check-label", default = False, action = "store_true",
      help="Check that there is no duplicated name/label.")

  args = parser.parse_args()
  root_dir = args.root
  list_file = args.listfile
  out_dir = args.outdir
  example_dir = args.exampledir

  redo = args.redo
  anno_type = args.anno_type
  label_type = args.label_type
  backend = args.backend
  check_size = args.check_size
  encode_type = args.encode_type
  encoded = args.encoded
  gray = args.gray
  label_map_file = args.label_map_file
  min_dim = args.min_dim
  max_dim = args.max_dim
  resize_height = args.resize_height
  resize_width = args.resize_width
  shuffle = args.shuffle
  check_label = args.check_label

  # check if root directory exists
  if not os.path.exists(root_dir):
    print "root directory: {} does not exist".format(root_dir)
    sys.exit()
  # add "/" to root directory if needed
  if root_dir[-1] != "/":
    root_dir += "/"
  # check if list file exists
  if not os.path.exists(list_file):
    print "list file: {} does not exist".format(list_file)
    sys.exit()
  # check list file format is correct
  with open(list_file, "r") as lf:
    for line in lf.readlines():
      img_file, anno = line.strip("\n").split(" ")
      if not os.path.exists(root_dir + img_file):
        print "image file: {} does not exist".format(root_dir + img_file)
      if anno_type == "classification":
        if not anno.isdigit():
          print "annotation: {} is not an integer".format(anno)
      elif anno_type == "detection":
        if not os.path.exists(root_dir + anno):
          print "annofation file: {} does not exist".format(root_dir + anno)
          sys.exit()
      break
  # check if label map file exist
  if anno_type == "detection":
    if not os.path.exists(label_map_file):
      print "label map file: {} does not exist".format(label_map_file)
      sys.exit()
    label_map = caffe_pb2.LabelMap()
    lmf = open(label_map_file, "r")
    try:
      text_format.Merge(str(lmf.read()), label_map)
    except:
      print "Cannot parse label map file: {}".format(label_map_file)
      sys.exit()
  out_parent_dir = os.path.dirname(out_dir)
  if not os.path.exists(out_parent_dir):
    os.makedirs(out_parent_dir)
  if os.path.exists(out_dir) and not redo:
    print "{} already exists and I do not hear redo".format(out_dir)
    sys.exit()
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

  # get caffe root directory
  caffe_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  if anno_type == "detection":
    cmd = "{}/build/tools/convert_annoset" \
        " --anno_type={}" \
        " --label_type={}" \
        " --label_map_file={}" \
        " --check_label={}" \
        " --min_dim={}" \
        " --max_dim={}" \
        " --resize_height={}" \
        " --resize_width={}" \
        " --backend={}" \
        " --shuffle={}" \
        " --check_size={}" \
        " --encode_type={}" \
        " --encoded={}" \
        " --gray={}" \
        " {} {} {}" \
        .format(caffe_root, anno_type, label_type, label_map_file, check_label,
            min_dim, max_dim, resize_height, resize_width, backend, shuffle,
            check_size, encode_type, encoded, gray, root_dir, list_file, out_dir)
  elif anno_type == "classification":
    cmd = "{}/build/tools/convert_annoset" \
        " --anno_type={}" \
        " --min_dim={}" \
        " --max_dim={}" \
        " --resize_height={}" \
        " --resize_width={}" \
        " --backend={}" \
        " --shuffle={}" \
        " --check_size={}" \
        " --encode_type={}" \
        " --encoded={}" \
        " --gray={}" \
        " {} {} {}" \
        .format(caffe_root, anno_type, min_dim, max_dim, resize_height,
            resize_width, backend, shuffle, check_size, encode_type, encoded,
            gray, root_dir, list_file, out_dir)
  print cmd
  process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
  output = process.communicate()[0]

  if not os.path.exists(example_dir):
    os.makedirs(example_dir)
  link_dir = os.path.join(example_dir, os.path.basename(out_dir))
  if os.path.exists(link_dir):
    os.unlink(link_dir)
  os.symlink(out_dir, link_dir)
