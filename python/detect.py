#!/usr/bin/env python
"""
detector.py is an out-of-the-box windowed detector
callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- come up with a batching scheme that preserved order / keeps a unique ID
"""
import numpy as np
import pandas as pd
import os
import argparse
import time

import caffe

CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
        "input_file",
        help="Input txt/csv filename. If .txt, must be list of filenames.\
        If .csv, must be comma-separated file with header\
        'filename, xmin, ymin, xmax, ymax'"
    )
    parser.add_argument(
        "output_file",
        help="Output h5/csv filename. Format depends on extension."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../examples/imagenet/imagenet_deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../examples/imagenet/caffe_reference_imagenet_model"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--crop_mode",
        default="center_only",
        choices=CROP_MODES,
        help="Image crop mode"
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=255,
        help="Multiply input features by this scale before input to net"
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    args = parser.parse_args()

    channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make detector.
    detector = caffe.Detector(args.model_def, args.pretrained_model,
            gpu=args.gpu, mean_file=args.mean_file,
            input_scale=args.input_scale, channel_swap=channel_swap)

    if args.gpu:
        print 'GPU mode'

    # Load input.
    t = time.time()
    print('Loading input...')
    if args.input_file.lower().endswith('txt'):
        with open(args.input_file) as f:
            inputs = [_.strip() for _ in f.readlines()]
    elif args.input_file.lower().endswith('csv'):
        inputs = pd.read_csv(args.input_file, sep=',', dtype={'filename': str})
        inputs.set_index('filename', inplace=True)
    else:
        raise Exception("Unknown input file type: not in txt or csv.")

    # Detect.
    if args.crop_mode == 'list':
        # Unpack sequence of (image filename, windows).
        images_windows = (
            (ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values)
            for ix in inputs.index.unique()
        )
        detections = detector.detect_windows(images_windows)
    else:
        detections = detector.detect_selective_search(inputs)
    print("Processed {} windows in {:.3f} s.".format(len(detections),
                                                     time.time() - t))

    # Collect into dataframe with labeled fields.
    df = pd.DataFrame(detections)
    df.set_index('filename', inplace=True)
    df[COORD_COLS] = pd.DataFrame(
        data=np.vstack(df['window']), index=df.index, columns=COORD_COLS)
    del(df['window'])

    # Save results.
    t = time.time()
    if args.output_file.lower().endswith('csv'):
        # csv
        # Enumerate the class probabilities.
        class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]
        df[class_cols] = pd.DataFrame(
            data=np.vstack(df['feat']), index=df.index, columns=class_cols)
        df.to_csv(args.output_file, cols=COORD_COLS + class_cols)
    else:
        # h5
        df.to_hdf(args.output_file, 'df', mode='w')
    print("Saved to {} in {:.3f} s.".format(args.output_file,
                                            time.time() - t))


if __name__ == "__main__":
    import sys
    main(sys.argv)
