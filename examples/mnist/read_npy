#!/usr/bin/env python

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Pretty-prints content of any .npy file')
parser.add_argument("input_file",help='the numpy data file you want to read')
args = parser.parse_args()

np.set_printoptions(precision=2)

print(np.load(args.input_file))

