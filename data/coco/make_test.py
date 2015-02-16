#!/usr/bin/env python

# This file is only meant to be run as a script with 0 arguments,
# and depends on steps 1-3 of README.md.
#
# It creates a test set from the image filenames of the test set.

import json
import os
import re

# get path to directory where this script is
script_dir = os.path.dirname(os.path.realpath(__file__))

set_name = 'test2014'
image_root = '%s/coco/images/%s' % (script_dir, set_name)
out_filename = '%s/coco/annotations/captions_%s.json' % (script_dir, set_name)
image_ext = 'jpg'
imname_re = re.compile('COCO_%s_(?P<image_id>\d+)\.%s' % (set_name, image_ext))
full_image_ext = '.%s' % image_ext
image_filenames = filter(lambda f: f.endswith(full_image_ext), os.listdir(image_root))
print 'Creating dummy annotation file for %d images at: %s' % \
    (len(image_filenames), out_filename)

out_data = {'type': 'captions', 'images': [], 'annotations': [],
            'licenses': [], 'info': {}}
for index, filename in enumerate(image_filenames):
    match = imname_re.match(filename)
    if match is None: raise Exception('Unsupported filename: %s' % filename)
    image_id = int(match.group('image_id'))
    out_data['images'].append({'file_name': filename, 'id': image_id})
    for dummy_index in range(2):
        annotation = {'caption': 'dummy caption %d' % dummy_index,
                      'id': index, 'image_id': image_id}
        out_data['annotations'].append(annotation)
with open(out_filename, 'w') as out_file:
    json.dump(out_data, out_file)
