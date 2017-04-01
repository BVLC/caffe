function solver = get_solver(solver_file)
% solver = get_solver(solver_file)
%   Construct a Solver object from solver_file

CHECK(ischar(solver_file), 'solver_file must be a string');
CHECK_FILE_EXIST(solver_file);
pSolver = caffe_('get_solver', solver_file);
if ~isstruct(pSolver)   % pSolver should be a handle (struct)
    error('null returned in get_solver');
else
    solver = caffe.Solver(pSolver);
end

end
