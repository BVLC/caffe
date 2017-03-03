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
"""
Form a subset of the Flickr Style data, download images to dirname, and write
Caffe ImagesDataLayer training file.
"""
import os
import urllib
import hashlib
import argparse
import numpy as np
import pandas as pd
from skimage import io
import multiprocessing

# Flickr returns a special image if the request is unavailable.
MISSING_IMAGE_SHA1 = '6a92790b1c2a301c6e7ddef645dca1f53ea97ac2'

example_dirname = os.path.abspath(os.path.dirname(__file__))
caffe_dirname = os.path.abspath(os.path.join(example_dirname, '../..'))
training_dirname = os.path.join(caffe_dirname, 'data/flickr_style')


def download_image(args_tuple):
    "For use with multiprocessing map. Returns filename on fail."
    try:
        url, filename = args_tuple
        if not os.path.exists(filename):
            urllib.urlretrieve(url, filename)
        with open(filename) as f:
            assert hashlib.sha1(f.read()).hexdigest() != MISSING_IMAGE_SHA1
        test_read_image = io.imread(filename)
        return True
    except KeyboardInterrupt:
        raise Exception()  # multiprocessing doesn't catch keyboard exceptions
    except:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download a subset of Flickr Style to a directory')
    parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help="random seed")
    parser.add_argument(
        '-i', '--images', type=int, default=-1,
        help="number of images to use (-1 for all [default])",
    )
    parser.add_argument(
        '-w', '--workers', type=int, default=-1,
        help="num workers used to download images. -x uses (all - x) cores [-1 default]."
    )
    parser.add_argument(
        '-l', '--labels', type=int, default=0,
        help="if set to a positive value, only sample images from the first number of labels."
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Read data, shuffle order, and subsample.
    csv_filename = os.path.join(example_dirname, 'flickr_style.csv.gz')
    df = pd.read_csv(csv_filename, index_col=0, compression='gzip')
    df = df.iloc[np.random.permutation(df.shape[0])]
    if args.labels > 0:
        df = df.loc[df['label'] < args.labels]
    if args.images > 0 and args.images < df.shape[0]:
        df = df.iloc[:args.images]

    # Make directory for images and get local filenames.
    if training_dirname is None:
        training_dirname = os.path.join(caffe_dirname, 'data/flickr_style')
    images_dirname = os.path.join(training_dirname, 'images')
    if not os.path.exists(images_dirname):
        os.makedirs(images_dirname)
    df['image_filename'] = [
        os.path.join(images_dirname, _.split('/')[-1]) for _ in df['image_url']
    ]

    # Download images.
    num_workers = args.workers
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count() + num_workers
    print('Downloading {} images with {} workers...'.format(
        df.shape[0], num_workers))
    pool = multiprocessing.Pool(processes=num_workers)
    map_args = zip(df['image_url'], df['image_filename'])
    results = pool.map(download_image, map_args)

    # Only keep rows with valid images, and write out training file lists.
    df = df[results]
    for split in ['train', 'test']:
        split_df = df[df['_split'] == split]
        filename = os.path.join(training_dirname, '{}.txt'.format(split))
        split_df[['image_filename', 'label']].to_csv(
            filename, sep=' ', header=None, index=None)
    print('Writing train/val for {} successfully downloaded images.'.format(
        df.shape[0]))
