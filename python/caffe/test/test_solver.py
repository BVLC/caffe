import unittest
import tempfile
import os
import numpy as np

import caffe
from test_net import simple_net_file

class TestSolver(unittest.TestCase):
    def setUp(self):
        self.num_output = 13
        net_f = simple_net_file(self.num_output)
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write("""net: '""" + net_f + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false""")
        f.close()
        self.solver = caffe.SGDSolver(f.name)
        # also make sure get_solver runs
        caffe.get_solver(f.name)
        caffe.set_mode_cpu()
        # fill in valid labels
        self.solver.net.blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.solver.net.blobs['label'].data.shape)
        self.solver.test_nets[0].blobs['label'].data[...] = \
                np.random.randint(self.num_output,
                    size=self.solver.test_nets[0].blobs['label'].data.shape)
        os.remove(f.name)
        os.remove(net_f)

    def test_solve(self):
        self.assertEqual(self.solver.iter, 0)
        self.solver.solve()
        self.assertEqual(self.solver.iter, 100)

    def test_net_memory(self):
        """Check that nets survive after the solver is destroyed."""

        nets = [self.solver.net] + list(self.solver.test_nets)
        self.assertEqual(len(nets), 2)
        del self.solver

        total = 0
        for net in nets:
            for ps in net.params.itervalues():
                for p in ps:
                    total += p.data.sum() + p.diff.sum()
            for bl in net.blobs.itervalues():
                total += bl.data.sum() + bl.diff.sum()
