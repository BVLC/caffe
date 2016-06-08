# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:32:59 2016

@author: denitome
"""

import re
import matplotlib.pyplot as plt

def parse_log(path_to_log):
    """Parse log file"""

    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile('Train net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile('Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    #regex_learning_rate = re.compile('lr = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')
    regex_loss = re.compile('loss = ([-+]?[0-9]*\.?[0-9]+([eE]?[-+]?[0-9]+)?)')

    # Pick out lines of interest
    iteration = -1
    train_data = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('stage',[])])
    test_data = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('stage',[])])

    with open(path_to_log) as f:
        
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                continue
            
            regex_loss_match = regex_loss.search(line)
            if regex_loss_match:
                loss_value = float(regex_loss_match.group(1))
                
            #learning_rate_match = regex_learning_rate.search(line)
            #if learning_rate_match:
            #    learning_rate = float(learning_rate_match.group(1))

            train_data = parse_line(regex_train_output, train_data, line, loss_value, iteration)
            test_data = parse_line(regex_test_output, test_data, line, loss_value, iteration)

    return train_data, test_data


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

def combine_data(train, test, new_train, new_test):
    last_iter = train['iteration'][-1]+ 1
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
    

def plotData(train, test, nstages):
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
    plt.semilogy(x,y,'r-', linewidth=2.0, label='Train')
    plt.grid()
    plt.xlabel('NUM ITERATIONS')
    plt.ylabel('LOSS')
    subtitle = 'Overall loss on all stages (min = %.4f; iter = %d)' % (min(y), x[y.index(min(y))])
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
    main_title = 'Training with\nbase_lr = %f; stage_1_lr_mul = %d; stage_n_lr_mul = %d' % (1e-5, 5, 1)
    plt.suptitle(main_title, size=14, fontweight='bold')
    plt.show()       
    

def main():
    filename = ['prototxt/log_tmp.txt','prototxt/log_tmp.txt']
    nstages = 6
    train, test = parse_log(filename[0])
    if (len(filename) > 1):
        for i in range(1,len(filename)):
            curr_tr, curr_ts = parse_log(filename[i])
            train, test = combine_data(train, test, curr_tr, curr_ts)
    plotData(train, test, nstages)

if __name__ == '__main__':
    main()
