function  net = tvg_matcaffe_init(use_gpu, gpu_id, model_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1
  error('Missing argument use_gpu');
end

if nargin < 2 || isempty(model_def_file)
  error('Missing argument model_def_file');
end

if nargin < 3 || isempty(model_file)
  error('Missing argument model_file');
end

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  fprintf('Using CPU Mode\n');
  caffe.set_mode_cpu();
end

if exist(model_file, 'file') ~= 2
    error('You need a network model file');
end
if exist(model_def_file,'file') ~= 2
    error('You need the network prototxt definition');
end
net = caffe.Net(model_def_file, model_file, 'test');
% put into test mode
fprintf('Done with set_phase_test\n');
fprintf('Done with init\n');

fprintf('Done with set_mode\n');

