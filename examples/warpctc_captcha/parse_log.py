#!/usr/bin/env python
# coding=utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

test_output = 'Test net output'
train_output = 'Train net output'

def main(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    accuracy, test_loss, train_loss = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].find(test_output) != -1:
            accuracy.append(float(lines[i][lines[i].rfind('=')+1:]))
            test_loss.append(float(lines[i+1][lines[i+1].rfind('=')+1:lines[i+1].rfind('loss')]))
            i += 2
        elif lines[i].find(train_output) != -1:
            train_loss.append(float(lines[i][lines[i].rfind('=')+1: lines[i].rfind('loss')]))
            i += 1
        else:
            i += 1

    # plot train loss
    plt.plot(train_loss)
    plt.title('train loss')
    plt.savefig('train_loss.png')
    plt.show()
    plt.close()
    # plot test loss
    plt.plot(test_loss)
    plt.title('test loss')
    plt.savefig('test_loss.png')
    plt.show()
    plt.close()
    # plt accracy
    plt.plot(accuracy)
    plt.title('test accuracy')
    plt.savefig('test_accuracy.png')
    plt.show()
    plt.close()

def print_help():
    print """this script do simple string match to parse train log file of warpctc demo
    
Usage:python parse_log.py input_log_file"""

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_help()
    else:
        main(sys.argv[1])
