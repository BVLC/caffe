from time import strftime
import traceback
import numpy as np
import os

import apollocaffe

class TrainLogger(object):
    def __init__(self, display_interval, log_file="/tmp/apollocaffe_log.txt"):
        self.display_interval = display_interval
        self.log_file = log_file
        os.system("touch %s" % self.log_file)
    def log(self, idx, meta_data):
        meta_data['start_iter'] = meta_data.get('start_iter', 0)
        if idx % self.display_interval == 0:
            log_line = ""
            try:
                loss = np.mean(meta_data['train_loss'][-self.display_interval:])
                log_line = "%s - Iteration %4d - Train Loss: %g" % \
                    (strftime("%Y-%m-%d %H:%M:%S"), idx, loss)
            except Exception as e:
                log_line += str(e)
                log_line = "Skipping training log: Unknown Error"

            try:
                with open(self.log_file, 'ab+') as lfile:
                    lfile.write("%s\n" % log_line)
            except IOError:
                print "Trainer Logger Error: %s does not exist." % self.log_file
            except Exception as e:
                print traceback.format_exc()
            print log_line

class TestLogger(object):
    def __init__(self, display_interval, log_file="/tmp/apollocaffe_log.txt"):
        self.display_interval = display_interval
        self.log_file = log_file
        os.system("touch %s" % self.log_file)
    def log(self, idx, meta_data):
        if idx % self.display_interval == 0:
            try:
                loss = meta_data['test_loss'][-1]
                log_line = "%s - Iteration %4d - Test Loss: %g" % \
                    (strftime("%Y-%m-%d %H:%M:%S"), idx, loss)
            except IndexError:
                log_line = "Skipping Test log: \
No test_loss provided"
            except Exception as e:
                log_line =  "Skipping test log: Unknown Error"
                print traceback.format_exc()

            try:
                with open(self.log_file, 'ab+') as lfile:
                    lfile.write("%s\n" % log_line)
            except IOError:
                print "TestLogger Error: %s does not exist." % self.log_file
            except Exception as e:
                print traceback.format_exc()
            print log_line

class SnapshotLogger(object):
    def __init__(self, snapshot_interval, snapshot_prefix='/tmp/model',
            log_file="/tmp/apollocaffe_log.txt"):
        self.snapshot_interval = snapshot_interval
        self.snapshot_prefix = snapshot_prefix
        self.log_file = log_file
        os.system("touch %s" % self.log_file)
    def log(self, idx, meta_data):
        meta_data['start_iter'] = meta_data.get('start_iter', 0)
        if idx % self.snapshot_interval == 0 and idx > meta_data['start_iter']:
            try:
                filename = '%s_%d.h5' % (self.snapshot_prefix, idx)
                log_line = "%s - Iteration %4d - Saving net to %s" % \
                    (strftime("%Y-%m-%d %H:%M:%S"), idx, filename)
                print(log_line)
                meta_data['apollo_net'].save(filename)
            except Exception as e:
                print traceback.format_exc()
                print('Saving failed')

class PlotLogger(object):
    def __init__(self, display_interval, plot_prefix='/tmp/apollo', boxcar_width={'train_loss': 100, 'test_loss': 1}):
        self.display_interval = display_interval
        self.plot_prefix = plot_prefix
        self.boxcar_width = boxcar_width
    def log(self, idx, meta_data):
        import matplotlib; matplotlib.use('Agg', warn=False); import matplotlib.pyplot as plt
        meta_data['start_iter'] = meta_data.get('start_iter', 0)
        if idx % self.display_interval == 0:
            try:
                for loss_type in ['train_loss', 'test_loss']:
                    history = meta_data[loss_type]
                    boxcar_width = self.boxcar_width[loss_type]
                    if len(history) < 2 * boxcar_width:
                        return
                    smoothed_loss = []
                    xaxis = []
                    for i in range(len(history) // boxcar_width):
                        smoothed_loss.append(np.mean(
                            history[i*boxcar_width:(i+1)*boxcar_width]))
                        step = int(boxcar_width * (idx - meta_data['start_iter']) / len(history))
                        xaxis.append(meta_data['start_iter'] + i * step)
                    plt.plot(xaxis, smoothed_loss)
                    plot_file = '%s_%s.jpg' % (self.plot_prefix, loss_type)
                    plt.savefig(plot_file)
                    plt.close()
                    print "%s - Iteration %4d - Saving %s plot to %s" % \
                        (strftime("%Y-%m-%d %H:%M:%S"), idx, loss_type, plot_file)
            except Exception as e:
                log_line = "Skipping train loss plot:"
                print traceback.format_exc()
