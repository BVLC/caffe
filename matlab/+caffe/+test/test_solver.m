% 
% All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
% 
% All contributions by the University of California:
% Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
% All rights reserved.
% 
% All other contributions:
% Copyright (c) 2014, 2015, the respective contributors
% All rights reserved.
% For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
% 
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
%     * Redistributions of source code must retain the above copyright notice,
%       this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%     * Neither the name of Intel Corporation nor the names of its contributors
%       may be used to endorse or promote products derived from this software
%       without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
classdef test_solver < matlab.unittest.TestCase
  
  properties
    num_output
    solver
  end
  
  methods
    function self = test_solver()
      self.num_output = 13;
      model_file = caffe.test.test_net.simple_net_file(self.num_output);
      solver_file = tempname();
      
      fid = fopen(solver_file, 'w');
      fprintf(fid, [ ...
        'net: "'  model_file  '"\n' ...
        'test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9\n' ...
        'weight_decay: 0.0005 lr_policy: "inv" gamma: 0.0001 power: 0.75\n' ...
        'display: 100 max_iter: 100 snapshot_after_train: false\n' ]);
      fclose(fid);
      
      self.solver = caffe.Solver(solver_file);
      % also make sure get_solver runs
      caffe.get_solver(solver_file);
      caffe.set_mode_cpu();
      % fill in valid labels
      self.solver.net.blobs('label').set_data(randi( ...
        self.num_output - 1, self.solver.net.blobs('label').shape));
      self.solver.test_nets(1).blobs('label').set_data(randi( ...
        self.num_output - 1, self.solver.test_nets(1).blobs('label').shape));
      
      delete(solver_file);
      delete(model_file);
    end
  end
  methods (Test)
    function test_solve(self)
      self.verifyEqual(self.solver.iter(), 0)
      self.solver.step(30);
      self.verifyEqual(self.solver.iter(), 30)
      self.solver.solve()
      self.verifyEqual(self.solver.iter(), 100)
    end
  end
end
