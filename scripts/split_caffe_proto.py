#!/usr/bin/env python
import mmap
import re
import os
import errno

script_path = os.path.dirname(os.path.realpath(__file__))

# a regex to match the parameter definitions in caffe.proto
r = re.compile(r'(?://.*\n)*message ([^ ]*) \{\n(?: .*\n|\n)*\}')

# create directory to put caffe.proto fragments
try:
    os.mkdir(
        os.path.join(script_path,
                     '../docs/_includes/'))
    os.mkdir(
        os.path.join(script_path,
                     '../docs/_includes/proto/'))
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

caffe_proto_fn = os.path.join(
    script_path,
    '../src/caffe/proto/caffe.proto')

with open(caffe_proto_fn, 'r') as fin:

    for m in r.finditer(fin.read(), re.MULTILINE):
        fn = os.path.join(
            script_path,
            '../docs/_includes/proto/%s.txt' % m.group(1))
        with open(fn, 'w') as fout:
            fout.write(m.group(0))
