#!/usr/bin/env python

# ##################################################################
# #                         DPENDENCES                             #
# ##################################################################

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np
import re, sys

def parse_log(log_file):
    with open(log_file, 'r') as log_file2:
        log = log_file2.read()

    # Train losses
    train_loss_pattern = r"Training:\tIteration = (?P<iter_num>\d+), \tloss = (?P<loss>[0-9]*[.][0-9]*)"
    train_loss_iterations = []
    train_losses = []

    for r in re.findall(train_loss_pattern, log):
        train_loss_iterations.append(int(r[0]))
        train_losses.append(float(r[1]))

    train_loss_iterations = np.array(train_loss_iterations)
    train_losses = np.array(train_losses)

    # Learning rate
    lr_pattern = r"Iteration (?P<iter_num>\d+), lr = (?P<lr>[0-9]*[.][0-9]*)"
    learning_rate = []

    for r in re.findall(lr_pattern, log):
        learning_rate.append(float(r[1]))

    learning_rate = np.array(learning_rate)


    # Test data
    test_pattern =  r"Testing:\tIteration = (?P<iter_num>\d+), \tloss = (?P<loss>[0-9]*[.][0-9]*), \taccuracy = (?P<acc>[0-9]*[.][0-9]*)"

    test_iterations = []
    test_accuracy = []
    test_losses = []

    for r in re.findall(test_pattern, log):
        test_iterations.append(int(r[0]))
        test_losses.append(float(r[1]))
        test_accuracy.append(float(r[2]))

    test_iterations = np.array(test_iterations)
    test_losses = np.array(test_losses)
    test_accuracy = np.array(test_accuracy)

    return train_loss_iterations, train_losses, test_iterations, test_losses, test_accuracy, learning_rate

if __name__ == '__main__':

    text_file = sys.argv[1]
    (train_loss_iterations, train_losses, test_iterations, test_losses, test_accuracy, learning_rate) = parse_log(text_file)

    plot_interval = 1
    train_loss_iterations = train_loss_iterations[0::plot_interval]
    train_losses = train_losses[0::plot_interval]

    max_acc     = max(test_accuracy)
    max_acc_idx = test_iterations[np.where(test_accuracy == max_acc)[0]][0]

    font = {'family' : 'cmr10', 'size'   : 16}
    plt.rc('font', **font)


    # Plot loss-accuracy curves
    host = host_subplot(111, axes_class=AA.Axes)

    host.clear()
    par1 = host.twinx()
    par2 = host.twinx()

    par1.axis["right"].toggle(all=True)
    par2.axis["right"].toggle(all=False)

    host.set_xlim(0, np.max(train_loss_iterations))
    host.set_ylim(0, np.max([np.max(train_losses), np.max(test_losses)]))



    host.set_title("Max Accuracy: {0:.4}% at iteration {1}".format(max_acc, max_acc_idx))
    host.set_xlabel("Iterations")
    host.set_ylabel("Loss")
    par1.set_ylabel("Accuracy (%)")

    p1, = host.plot(train_loss_iterations, train_losses, label="Train loss", linewidth=1.5)
    p2, = host.plot(test_iterations, test_losses, label="Test loss", linewidth=1.5)
    p3, = par1.plot(test_iterations, test_accuracy, label="Accuracy", linewidth=1.5)

    par1.set_ylim(0, 1)
    host.legend(loc='lower left', ncol=1, fancybox=False, shadow=True)
    # # Plot learning rate
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.plot(learning_rate, 'b-')
    # ax.set_ylim(0, np.max(learning_rate) + np.max(learning_rate)*0.01)
    # ax.set_xlim(0, len(learning_rate))
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('Learning rate')
    # ax.set_xticklabels(["$%d$"%f for f in np.arange(0, np.max(train_loss_iterations), len(train_loss_iterations))])
    # ax.set_title("Learning rate decay curve")

    plt.show()






















#
