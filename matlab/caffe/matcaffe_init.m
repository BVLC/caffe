function  matcaffe_init(varargin)
% Initilize matcaffe wrapper
% matcaffe_init(varargin)
% 
% PARAMETERS
% b_use_gpu
% model_def_file
% model_file
% b_force: if false (default) and model is already initialized, do not re-initialize; if
%  true, re-initialize regardless

% Modified by Daniel Golden (dan at cellscope dot com) August 2014

%% Parse input arguments
p = inputParser;
p.addParameter('b_use_gpu', false);
p.addParameter('model_def_file', fullfile(getenv('CAFFE_HOME'), 'examples/imagenet/imagenet_deploy.prototxt'));
p.addParameter('model_file', fullfile(getenv('CAFFE_HOME'), 'examples/imagenet/caffe_reference_imagenet_model'));
p.addParameter('b_force', false);
p.parse(varargin{:});

%% Go
if p.Results.b_force || caffe('is_initialized') == 0
  if exist(p.Results.model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(p.Results.model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
  end
  caffe('init', p.Results.model_def_file, p.Results.model_file)
end
fprintf('Done with init\n');

% set to use GPU or CPU
if p.Results.b_use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');

% put into test mode
caffe('set_phase_test');
fprintf('Done with set_phase_test\n');
