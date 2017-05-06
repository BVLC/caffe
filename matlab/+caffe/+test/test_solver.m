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
