#!/usr/bin/env python

# This file is only meant to be run as a script with 0 arguments,
# and depends on steps 1-3 of README.md.
#
# It creates a "trainval" set by combining the COCO 2014 train and val sets.
# The trainval set is intended for use only when training a single final model
# for submission of results on the test set to the COCO evaluation server.

import os
import json

# get path to directory where this script is
script_dir = os.path.dirname(os.path.realpath(__file__))

anno_dir_path = '%s/coco/annotations' % script_dir
image_root = '%s/coco/images' % script_dir
abs_image_root = os.path.abspath(image_root)
out_coco_id_filename = '%s/coco2014_cocoid.trainval.txt' % script_dir
filename_pattern = 'captions_%s2014.json'
in_sets = ['train', 'val']
out_set = 'trainval'
path_pattern = '%s/%s' % (anno_dir_path, filename_pattern)

out_data = {}
for in_set in in_sets:
    filename = path_pattern % in_set
    print 'Loading input dataset from: %s' % filename
    data = json.load(open(filename, 'r'))
    for key, val in data.iteritems():
        if type(val) == list:
            if key not in out_data:
                out_data[key] = []
            out_data[key] += val
        else:
            if key not in out_data:
                out_data[key] = val
            assert out_data[key] == val
filename = path_pattern % out_set
print 'Dumping output dataset to: %s' % filename
json.dump(out_data, open(filename, 'w'))

out_ids = [str(im['id']) for im in out_data['images']]
print 'Writing COCO IDs to: %s' % out_coco_id_filename
with open(out_coco_id_filename, 'w') as coco_id_file:
    coco_id_file.write('\n'.join(out_ids) + '\n')

# make a trainval dir with symlinks to all train+val images
out_dir = '%s/%s2014' % (image_root, out_set)
os.makedirs(out_dir)
print 'Writing image symlinks to: %s' % out_dir
for im in out_data['images']:
    filename = im['file_name']
    set_name = None
    for in_set in in_sets:
        if in_set in filename:
            set_name = in_set
            break
    assert set_name is not None
    real_path = '%s/%s2014/%s' % (abs_image_root, set_name, filename)
    link_path = '%s/%s' % (out_dir, filename)
    os.symlink(real_path, link_path)
