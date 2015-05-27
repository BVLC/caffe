function results = run_tests()
% results = run_tests()
%   run all tests in this caffe matlab wrapper package

caffe.reset();
results = [...
  run(caffe.test.test_net) ...
  run(caffe.test.test_solver)
  ];
caffe.reset();

end
