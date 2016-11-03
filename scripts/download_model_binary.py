#!/usr/bin/env python
# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import sys
import time
import yaml
import urllib
import hashlib
import argparse

required_keys = ['caffemodel', 'caffemodel_url', 'sha1']


def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = (time.time() - start_time) or 0.01
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def parse_readme_frontmatter(dirname):
    readme_filename = os.path.join(dirname, 'readme.md')
    with open(readme_filename) as f:
        lines = [line.strip() for line in f.readlines()]
    top = lines.index('---')
    bottom = lines.index('---', top + 1)
    frontmatter = yaml.load('\n'.join(lines[top + 1:bottom]))
    assert all(key in frontmatter for key in required_keys)
    return dirname, frontmatter


def valid_dirname(dirname):
    try:
        return parse_readme_frontmatter(dirname)
    except Exception as e:
        print('ERROR: {}'.format(e))
        raise argparse.ArgumentTypeError(
            'Must be valid Caffe model directory with a correct readme.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download trained model binary.')
    parser.add_argument('dirname', type=valid_dirname)
    args = parser.parse_args()

    # A tiny hack: the dirname validator also returns readme YAML frontmatter.
    dirname = args.dirname[0]
    frontmatter = args.dirname[1]
    model_filename = os.path.join(dirname, frontmatter['caffemodel'])

    # Closure-d function for checking SHA1.
    def model_checks_out(filename=model_filename, sha1=frontmatter['sha1']):
        with open(filename, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest() == sha1

    # Check if model exists.
    if os.path.exists(model_filename) and model_checks_out():
        print("Model already exists.")
        sys.exit(0)

    # Download and verify model.
    urllib.urlretrieve(
        frontmatter['caffemodel_url'], model_filename, reporthook)
    if not model_checks_out():
        print('ERROR: model did not download correctly! Run this again.')
        sys.exit(1)
