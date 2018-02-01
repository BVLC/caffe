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

import matplotlib as mpl
mpl.use('Agg')
import os
import sys
import re
import random
import matplotlib.pyplot as plt


def cut_multi_log(path_to_multi_log):
    logfile_dict = {}
    multi_flag = False
    pattern = re.compile('\[\d\]')
    with open(path_to_multi_log,'r') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                multi_flag = True
                num = m.group()[1]
                if num in logfile_dict.keys():
                    file_data = logfile_dict[num]
                    file_data.append(line[3:])
                else:
                    file_data = []
                    file_data.append(line[3:])
                    logfile_dict[num] = file_data

        log_name_list = []
        for key in logfile_dict.keys():
            log_base_name = os.path.basename(path_to_multi_log)
            new_log_file_name = log_base_name + '-'+key
            file_data=logfile_dict[key]
            log_name_list.append(new_log_file_name)
            with open(new_log_file_name,'w') as f:
                for item in file_data:
                    f.write(item)

    if multi_flag == True:
        return log_name_list
    else:
        return [path_to_multi_log]

def lr_exist(filename):
    with open(filename,'r') as f:
        for line in f:
            line = line.strip()
            if line[0] != '#':
                line_list = line.split()
                if len(line_list) < 4:
                    return False
                else:
                    return True

def plot_train_graph(logpath,png_path):
    get_data(logpath)
    basename = os.path.basename(logpath)
    train_file = basename+'.train'
    loss_data = get_plot_data(train_file,0,2)
    linewidth = 0.75
    color = [random.random(), random.random(), random.random()]

    plt.plot(loss_data[0], loss_data[1], label = 'Loss', color = color,marker = 'o', linewidth = linewidth,markersize=2)
    plt.title('Training Graph: Iter vs Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    loss_path = os.path.join(png_path,basename+'.loss.png')
    plt.savefig(loss_path)
    plt.show()

    if not lr_exist(train_file):
        print "no learning rate data"
        return

    LR_data = get_plot_data(train_file,0,3)
    plt.clf()
    color = [random.random(), random.random(), random.random()]
    plt.plot(LR_data[0], LR_data[1], label = 'Learning Rate', color = color,marker = 'o', linewidth = linewidth,markersize=2)
    plt.title('Training Graph: Iter vs Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate')
    lr_path = os.path.join(png_path,basename+'.lr.png')
    plt.savefig(lr_path)
    plt.show()

def plot_test_graph(logpath,png_path):
    basename = os.path.basename(logpath)
    get_data(logpath)
    test_file = basename+'.test'
    acc_data = get_plot_data(test_file,0,2)
    linewidth = 0.75
    color = [random.random(), random.random(), random.random()]
    plt.clf()
    plt.plot(acc_data[0], acc_data[1], label = 'accuracy', color = color,marker = 'o', linewidth = linewidth)
    plt.title('Test Graph: Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('acc')
    test_path = os.path.join(png_path,basename+'.acc.png')
    plt.savefig(test_path)
    plt.show()

def get_plot_data(filepath,idx1,idx2):
    data = [[],[]]
    base_line_flag = True
    base_length = 4
    with open(filepath,'r') as f:
        for line in f:
            line = line.strip()
            if line[0] != '#':
                line_list = line.split()
                if base_line_flag == True:
                    base_line_flag = False
                    base_length = len(line_list)

                if len(line_list) < base_length:
                    return data
                data[0].append(float(line_list[idx1].strip()))
                data[1].append(float(line_list[idx2].strip()))
    return data

def get_data(logpath):
    sh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'parse_log.sh')
    if os.system('{} {}'.format(sh_path, logpath)):
        sys.exit(1)


