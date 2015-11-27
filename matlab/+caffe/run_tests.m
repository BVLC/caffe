function results = run_tests()
% results = run_tests()
%   run all tests in this caffe matlab wrapper package

% use CPU for testing
caffe.set_mode_cpu();

% reset caffe before testing
caffe.reset_all();

% put all test cases here
results = [...
  run(caffe.test.test_net) ...
  run(caffe.test.test_solver) ...
  run(caffe.test.test_io) ];

% reset caffe after testing
caffe.reset_all();

end
