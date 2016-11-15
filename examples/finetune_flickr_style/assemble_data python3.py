#!/usr/bin/env python
"""
For python3 assemble_data usage same with assemble_data.py 
"""
"""
Form a subset of the Flickr Style data, download images to dirname, and write
Caffe ImagesDataLayer training file.
"""
import os
import urllib.request
import hashlib
import argparse
import numpy as np
import pandas as pd
from skimage import io
import multiprocessing
import imghdr
# Flickr returns a special image if the request is unavailable.
example_dirname = os.path.abspath(os.path.dirname(__file__))
caffe_dirname = os.path.abspath(os.path.join(example_dirname, '../..'))
training_dirname = os.path.join(caffe_dirname, 'data/flickr_style')


def download_image(args_tuple):
    "For use with multiprocessing map. Returns filename on fail."
    # try:
    url, filename = args_tuple
        
    if not os.path.exists(filename):
        while True: # urllib will fail if unstable Internet Connection
            try:
                finalUrl = urllib.request.urlopen(url).geturl()
                break
            except:
                pass
        print("Start: ---" + finalUrl, filename)
        if finalUrl.find("unavailable") != -1: # postfix of url is "photo_unavailable.png"
            print("Final Url Test Failed!")
            return False

        try:
            # put request here to avoid urlError
            urllib.request.urlretrieve(url, filename=filename)
            print("Success")
            return True
        except:
            print("urlretrieve API Test Failed!")
            return False
    print("exist: ", filename)
    return True


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
    df = pd.read_csv(csv_filename, index_col=0, compression='gzip', encoding='utf-8')
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


