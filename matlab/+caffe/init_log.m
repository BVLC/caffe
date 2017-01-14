function init_log(log_base_filename)
% init_log(log_base_filename)
%   init Caffe's log

CHECK(ischar(log_base_filename) && ~isempty(log_base_filename), ...
  'log_base_filename must be string');

[log_base_dir] = fileparts(log_base_filename);
if ~exist(log_base_dir, 'dir')
    mkdir(log_base_dir);
end

caffe_('init_log', log_base_filename);

end
