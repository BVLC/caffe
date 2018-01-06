function solver = get_solver(solver_file)
% solver = get_solver(solver_file)
%   Construct a Solver object from solver_file

CHECK(ischar(solver_file), 'solver_file must be a string');
CHECK_FILE_EXIST(solver_file);
pSolver = caffe_('get_solver', solver_file);
solver = caffe.Solver(pSolver);

end
