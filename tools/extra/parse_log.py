#!/usr/bin/env python

"""
Parse training log

Competitor to parse_log.sh
"""

import os
import re
import extract_seconds
import argparse
import csv


def get_line_type(line):
    """Return either 'test' or 'train' depending on line type
    """

    line_type = None
    if line.find('Train') != -1:
        line_type = 'train'
    elif line.find('Test') != -1:
        line_type = 'test'
    return line_type


def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, train_dict_names, test_dict_list, test_dict_names)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows

    train_dict_names and test_dict_names are ordered tuples of the column names
    for the two dict_lists
    """

    re_iteration = re.compile('Iteration (\d+)')
    re_accuracy = re.compile('output #\d+: accuracy = ([\.\d]+)')
    re_train_loss = re.compile('Iteration \d+, loss = ([\.\d]+)')
    re_output_loss = re.compile('output #\d+: loss = ([\.\d]+)')
    re_lr = re.compile('lr = ([\.\d]+)')

    # Pick out lines of interest
    iteration = -1
    test_accuracy = -1
    learning_rate = float('NaN')
    train_dict_list = []
    test_dict_list = []
    train_dict_names = ('NumIters', 'Seconds', 'TrainingLoss', 'LearningRate')
    test_dict_names = ('NumIters', 'Seconds', 'TestAccuracy', 'TestLoss')

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)

        for line in f:
            iteration_match = re_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                # Only look for other stuff if we've found the first iteration
                continue

            time = extract_seconds.extract_datetime_from_line(line,
                                                              logfile_year)
            seconds = (time - start_time).total_seconds()

            lr_match = re_lr.search(line)
            if lr_match:
                learning_rate = float(lr_match.group(1))

            accuracy_match = re_accuracy.search(line)
            if accuracy_match and get_line_type(line) == 'test':
                test_accuracy = float(accuracy_match.group(1))

            train_loss_match = re_train_loss.search(line)
            if train_loss_match:
                train_loss = float(train_loss_match.group(1))
                train_dict_list.append({'NumIters': iteration,
                                        'Seconds': seconds,
                                        'TrainingLoss': train_loss,
                                        'LearningRate': learning_rate})

            output_loss_match = re_output_loss.search(line)
            if output_loss_match and get_line_type(line) == 'test':
                test_loss = float(output_loss_match.group(1))
                # NOTE: we assume that (1) accuracy always comes right before
                # loss for test data so the test_accuracy variable is already
                # correctly populated and (2) there's one and only one output
                # named "accuracy" for the test net
                test_dict_list.append({'NumIters': iteration,
                                       'Seconds': seconds,
                                       'TestAccuracy': test_accuracy,
                                       'TestLoss': test_loss})

    return train_dict_list, train_dict_names, test_dict_list, test_dict_names


def save_csv_files(logfile_path, output_dir, train_dict_list, train_dict_names,
                   test_dict_list, test_dict_names, verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    write_csv(train_filename, train_dict_list, train_dict_names, verbose)

    test_filename = os.path.join(output_dir, log_basename + '.test')
    write_csv(test_filename, test_dict_list, test_dict_names, verbose)


def write_csv(output_filename, dict_list, header_names, verbose=False):
    """Write a CSV file
    """

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, header_names)
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print 'Wrote %s' % output_filename


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dict_list, train_dict_names, test_dict_list, test_dict_names = \
        parse_log(args.logfile_path)
    save_csv_files(args.logfile_path, args.output_dir, train_dict_list,
                   train_dict_names, test_dict_list, test_dict_names)


if __name__ == '__main__':
    main()
