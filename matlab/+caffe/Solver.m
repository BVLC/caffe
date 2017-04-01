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
    function delete (self)
      caffe_('delete_solver', self.hSolver_self);
    end
    function iter = iter(self)
      iter = caffe_('solver_get_iter', self.hSolver_self);
    end
    function restore(self, snapshot_filename)
      CHECK(ischar(snapshot_filename), 'snapshot_filename must be a string');
      CHECK_FILE_EXIST(snapshot_filename);
      if caffe_('solver_restore', self.hSolver_self, snapshot_filename) == -1
          error('null returned in caffe.Solver.restore(snapshot_filename)');
      end
    end
    function solve(self)
      if caffe_('solver_solve', self.hSolver_self) == -1
          error('null returned in caffe.Solver.solve()');
      end
    end
    function step(self, iters)
      CHECK(isscalar(iters) && iters > 0, 'iters must be positive integer');
      iters = double(iters);
      if caffe_('solver_step', self.hSolver_self, iters) == -1
          error('null returned in caffe.Solver.step()');
      end
    end
  end
end
