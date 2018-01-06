#!/usr/bin/python
#
# copyright Guillaume Dumont (2016)

import os
import sys
import hashlib
import argparse
import tarfile

from six.moves import urllib
from download_model_binary import reporthook

WIN_DEPENDENCIES_URLS = {
    ('v120', '2.7'):("https://github.com/willyd/caffe-builder/releases/download/v1.1.0/libraries_v120_x64_py27_1.1.0.tar.bz2",
                  "ba833d86d19b162a04d68b09b06df5e0dad947d4"),
    ('v140', '2.7'):("https://github.com/willyd/caffe-builder/releases/download/v1.1.0/libraries_v140_x64_py27_1.1.0.tar.bz2",
                  "17eecb095bd3b0774a87a38624a77ce35e497cd2"),
    ('v140', '3.5'):("https://github.com/willyd/caffe-builder/releases/download/v1.1.0/libraries_v140_x64_py35_1.1.0.tar.bz2",
                  "f060403fd1a7448d866d27c0e5b7dced39c0a607"),
}

# function for checking SHA1.
def model_checks_out(filename, sha1):
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download prebuilt dependencies for windows.')
    parser.add_argument('--msvc_version', default='v120', choices=['v120', 'v140'])
    args = parser.parse_args()

    # get the appropriate url
    pyver = '{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor)
    try:
        url, sha1 = WIN_DEPENDENCIES_URLS[(args.msvc_version, pyver)]
    except KeyError:
        print('ERROR: Could not find url for MSVC version = {} and Python version = {}.\n{}'
              .format(args.msvc_version, pyver,
              'Available combinations are: {}'.format(list(WIN_DEPENDENCIES_URLS.keys()))))
        sys.exit(1)

    dep_filename = os.path.split(url)[1]
    # Download binaries
    print("Downloading dependencies ({}). Please wait...".format(dep_filename))
    urllib.request.urlretrieve(url, dep_filename, reporthook)
    if not model_checks_out(dep_filename, sha1):
        print('ERROR: dependencies did not download correctly! Run this again.')
        sys.exit(1)
    print("\nDone.")

    # Extract the binaries from the tar file
    tar = tarfile.open(dep_filename, 'r:bz2')
    print("Extracting dependencies. Please wait...")
    tar.extractall()
    print("Done.")
    tar.close()

