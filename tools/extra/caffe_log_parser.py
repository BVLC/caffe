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
import re
from DFA_multi_parser_time import time_DFA
import argparse
import log_plot

parser = argparse.ArgumentParser(description='caffe log parser for training time and plot.')
parser.add_argument('--time',default = True, action = 'store_true')
parser.add_argument('log_file',  help='caffe training log file')
parser.add_argument('--out_dir',help='relavent file output path',default='./img_out')
parser.add_argument('--plot', default = True, help=' --plot draw graph( Iter vs LR, Iter vs Loss and Iter vs acc) \n ', action = 'store_true')
args = parser.parse_args()


def cut_multi_log(path_to_multi_log):
    multi_flag = False
    pattern = re.compile('^\[0\]')
    file_content = []
    with open(path_to_multi_log,'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                file_content.append(line[3:])
                multi_flag = True
    if multi_flag:
        log_base_name = os.path.basename(path_to_multi_log)
        cuted_log_path = os.path.join('/tmp/', log_base_name+'-0')
        with open(cuted_log_path, 'w') as f:
            for line in file_content:
                f.write(line)
            f.close()
        return cuted_log_path
    else:
        return path_to_multi_log


def main():
    log_path = args.log_file
    out_dir = args.out_dir
    log_path_cuted = cut_multi_log(log_path)
    if args.time:
        test = time_DFA(log_path_cuted)
        test.parse_time()
        print 'Total time(hour) : ',test.total_time/60.0/60.0
        print 'Total test time(hour) :',test.total_test_time/60.0/60.0
        print 'Total snapshot time(hour) : ',test.total_snapshot_time/60.0/60.0
        print 'Total train time(hour) : ',(test.total_time - test.total_test_time - test.total_snapshot_time)/60.0/60.0

    if args.plot:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        log_plot.plot_train_graph(log_path_cuted, out_dir)
        log_plot.plot_test_graph(log_path_cuted, out_dir)
        print 'Plot train/test graph saved at : ', out_dir


if __name__ == '__main__':
	main()


