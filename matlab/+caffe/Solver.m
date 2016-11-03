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
classdef Solver < handle
  % Wrapper class of caffe::SGDSolver in matlab
  
  properties (Access = private)
    hSolver_self
    attributes
    % attribute fields
    %     hNet_net
    %     hNet_test_nets
  end
  properties (SetAccess = private)
    net
    test_nets
  end
  
  methods
    function self = Solver(varargin)
      % decide whether to construct a solver from solver_file or handle
      if ~(nargin == 1 && isstruct(varargin{1}))
        % construct a solver from solver_file
        self = caffe.get_solver(varargin{:});
        return
      end
      % construct a solver from handle
      hSolver_solver = varargin{1};
      CHECK(is_valid_handle(hSolver_solver), 'invalid Solver handle');
      
      % setup self handle and attributes
      self.hSolver_self = hSolver_solver;
      self.attributes = caffe_('solver_get_attr', self.hSolver_self);
      
      % setup net and test_nets
      self.net = caffe.Net(self.attributes.hNet_net);
      self.test_nets = caffe.Net.empty();
      for n = 1:length(self.attributes.hNet_test_nets)
        self.test_nets(n) = caffe.Net(self.attributes.hNet_test_nets(n));
      end
    end
    function iter = iter(self)
      iter = caffe_('solver_get_iter', self.hSolver_self);
    end
    function restore(self, snapshot_filename)
      CHECK(ischar(snapshot_filename), 'snapshot_filename must be a string');
      CHECK_FILE_EXIST(snapshot_filename);
      caffe_('solver_restore', self.hSolver_self, snapshot_filename);
    end
    function solve(self)
      caffe_('solver_solve', self.hSolver_self);
    end
    function step(self, iters)
      CHECK(isscalar(iters) && iters > 0, 'iters must be positive integer');
      iters = double(iters);
      caffe_('solver_step', self.hSolver_self, iters);
    end
  end
end
