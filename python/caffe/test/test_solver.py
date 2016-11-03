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
import unittest
import tempfile
import os
import numpy as np
import six

import caffe
from test_net import simple_net_file


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.num_output = 13
        net_f = simple_net_file(self.num_output)
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write("""net: '""" + net_f + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false
        snapshot_prefix: "model" """)
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
            for ps in six.itervalues(net.params):
                for p in ps:
                    total += p.data.sum() + p.diff.sum()
            for bl in six.itervalues(net.blobs):
                total += bl.data.sum() + bl.diff.sum()

    def test_snapshot(self):
        self.solver.snapshot()
        # Check that these files exist and then remove them
        files = ['model_iter_0.caffemodel', 'model_iter_0.solverstate']
        for fn in files:
            assert os.path.isfile(fn)
            os.remove(fn)
