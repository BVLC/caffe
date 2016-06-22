# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:32:59 2016

@author: denitome
"""

import re
import matplotlib.pyplot as plt
import stat
from mpldatacursor import datacursor

def parse_log(path_to_log):
    """Parse log file"""

    regex_base_lr = re.compile('base_lr: ([\.\deE+-]+)')
    regex_stepsize = re.compile('stepsize: (\d+)')
    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    #regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')
    regex_loss = re.compile('loss = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    base_lr = -1
    stepsize = -1
    train_data = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('stage',[])])
    test_data = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('stage',[])])

    with open(path_to_log) as f:
        
        for line in f:
            if base_lr == -1:
                base_lr_match = regex_base_lr.search(line)
                if base_lr_match:
                    base_lr = float(base_lr_match.group(1))
            if stepsize == -1:
                stepsize_match = regex_stepsize.search(line)
                if stepsize_match:
                    stepsize = int(stepsize_match.group(1))
            
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                continue
            
            regex_loss_match = regex_loss.search(line)
            if regex_loss_match:
                loss_value = float(regex_loss_match.group(1))

            train_data = parse_line(regex_train_output, train_data, line, loss_value, iteration)
            test_data = parse_line(regex_test_output, test_data, line, loss_value, iteration)

    return train_data, test_data, base_lr, stepsize


def parse_line(regex_obj, data, line, loss_iter, iteration):
    """Parse a single line for training or test output"""

    output_match = regex_obj.search(line)
    if output_match:
        stage = int(output_match.group(1)) + 1
        loss_stage = float(output_match.group(3))
        data['iteration'].append(iteration)
        data['loss_iter'].append(loss_iter)
        data['loss_stage'].append(loss_stage)
        data['stage'].append(stage)
    return data

def combine_data(train, test, new_train, new_test, merge):
    if (merge == 0):
        last_iter = train['iteration'][-1]+ 1
    else:
        last_iter = merge
        iter_idx = train['iteration'].index(merge)
        for i in range(iter_idx,len(train['iteration'])):
            del train['iteration'][-1]
            del train['loss_iter'][-1]
            del train['loss_stage'][-1]
            del train['stage'][-1]
            
    if (new_train['iteration'][0] > 0):
        last_iter = 0
        
    for i in range(len(new_train['iteration'])):
        train['iteration'].append(last_iter + new_train['iteration'][i])
        train['loss_iter'].append(new_train['loss_iter'][i])
        train['loss_stage'].append(new_train['loss_stage'][i])
        train['stage'].append(new_train['stage'][i])
    if (len(test['iteration']) > 0):
        last_iter = test['iteration'][-1] + 1
        for i in range(len(new_test)):
            test['iteration'].append(last_iter + new_test['iteration'][i])
            test['loss_iter'].append(new_test['loss_iter'][i])
            test['loss_stage'].append(new_test['loss_stage'][i])
            test['stage'].append(new_test['stage'][i])
    return train, test

def smoothed_data(x, y, batch_size):
    smoothed_x = []
    smoothed_y = []
    smoothed_x.append(x[0])
    smoothed_y.append(y[0])
    i = 0
    for i in range(int(len(x)/batch_size)):
        curr_batch = y[i*batch_size:(i+1)*batch_size]
        smoothed_x.append(x[(i+1)*batch_size])
        smoothed_y.append(mean(curr_batch))
    # Consider elements left
    smoothed_x.append(x[-1])
    smoothed_y.append(mean(y[i*batch_size:]))
    return smoothed_x, smoothed_y
    
    

def plotData(train, test, nstages, main_title, avg_line = False, avg_batch_size = 5):
    x = []
    y = []
    count = 0
    for i in train['stage']:
        if i == 1:
            x.append(train['iteration'][count])
            y.append(train['loss_iter'][count])
        count += 1
    plt.clf()
    plt.subplot(211)
    subtitle = 'Overall loss on all stages (min = %.4f; iter = %d)' % (min(y), x[y.index(min(y))])
    plt.semilogy(x,y,'r-', linewidth=2.0, label='Train')
    if avg_line:
        x, y = smoothed_data(x, y, avg_batch_size)
        plt.semilogy(x,y,'b--', linewidth=2.0, label='Smoothed')
    plt.grid()
    plt.xlabel('NUM ITERATIONS')
    plt.ylabel('LOSS')
    plt.title(subtitle, fontweight='bold')
    plt.legend(loc='upper right')
    
    # plot loss per each stage
    x = [[] for i in range(6)]
    y = [[] for i in range(6)]
    count = 0
    for i in train['stage']:
        x[i-1].append(train['iteration'][count])
        y[i-1].append(train['loss_stage'][count])
        count += 1
    plt.subplot(212)
    plt.semilogy(x[0],y[0],linewidth=2.0,label='stage1')
    plt.semilogy(x[1],y[1],linewidth=2.0,label='stage2')
    plt.semilogy(x[2],y[2],linewidth=2.0,label='stage3')
    plt.semilogy(x[3],y[3],linewidth=2.0,label='stage4')
    plt.semilogy(x[4],y[4],linewidth=2.0,label='stage5')
    plt.semilogy(x[5],y[5],linewidth=2.0,label='stage6')
    plt.grid()
    plt.xlabel('NUM ITERATIONS')
    plt.ylabel('LOSS')
    min_val_stages = [min(y[idx_s]) for idx_s in range(nstages)]
    idx_min_stage = min_val_stages.index(min(min_val_stages)) + 1
    subtitle = 'Loss per stage (min = %0.4f; stage = %d)' % (min(min_val_stages), idx_min_stage)
    plt.title(subtitle, fontweight='bold')
    plt.legend(loc='upper right')
    plt.suptitle(main_title, size=14, fontweight='bold')
    datacursor(bbox=None, display='single', formatter="Iter:{x:.0f}\nLoss:{y:.2f}".format)
    plt.show()       
    

def main():
    #filename = ['prototxt/caffemodel/trial_3/log.txt']
    #filename = ['prototxt/caffemodel/trial_3/log.txt','prototxt/log.txt']
    filename = ['prototxt/log1.txt','prototxt/log.txt']
    stn_lrm = 1
    nstages = 6
    merge = [50000]
    #merge = [0]
    train, test, base_lr, stepsize = parse_log(filename[0])
    print 'Num iterations file = %d' % (train['iteration'][-1])
    if (len(filename) > 1):
        for i in range(1,len(filename)):
            curr_tr, curr_ts, base_lr_, stepsize_ = parse_log(filename[i])
            print 'Num iterations file = %d' % (curr_tr['iteration'][-1])
            train, test = combine_data(train, test, curr_tr, curr_ts, merge[i-1])
    main_title = 'Training with:\nbase_lr = %f; stepsize = %d; lr_mul = %d\nFinetuning: trial_2; Iter = %d ' % (base_lr, stepsize, stn_lrm, merge[0])
    plotData(train, test, nstages, main_title, avg_line = True, avg_batch_size = 500)

if __name__ == '__main__':
    main()
