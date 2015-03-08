% ------------------------------------------------------------------------
% net_surgery_demo
% ------------------------------------------------------------------------
caffe_root = '../../';
deploy_file = fullfile(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt');
model_file = fullfile(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel');
deploy_conv_file = fullfile(caffe_root, 'examples/imagenet/bvlc_caffenet_full_conv.prototxt');

if ~exist(model_file, 'file')
  fprintf('Please download caffe reference net first\n');
  return;
end

%% get weights in the net with fully-connected layers
caffe('reset');
caffe('init', deploy_file, model_file);
fc_weights = caffe('get_weights');
% print blob dimensions
fc_names = {'fc6', 'fc7', 'fc8'};
fc_layer_ids = zeros(3, 1);
for ii = 1:numel(fc_names)
  lname = fc_names{ii};
  for jj = 1:numel(fc_weights)
    if (strcmp(fc_weights(jj).layer_names, lname))
      fprintf('%s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        lname, sprintf('%d ', size(fc_weights(jj).weights{1})), ...
        sprintf('%d ', size(fc_weights(jj).weights{2})));
      fc_layer_ids(ii) = jj;
    end
  end
end

%% get weights in full-convolutional net
caffe('reset');
caffe('init', deploy_conv_file, model_file);
conv_weights = caffe('get_weights');
% print corresponding blob dimensions
conv_names = {'fc6-conv', 'fc7-conv', 'fc8-conv'};
conv_layer_ids = zeros(3, 1);
for ii = 1:numel(conv_names)
  lname = conv_names{ii};
  for jj = 1:numel(conv_weights)
    if (strcmp(conv_weights(jj).layer_names, lname))
      fprintf('%s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        lname, sprintf('%d ', size(conv_weights(jj).weights{1})), ...
        sprintf('%d ', size(conv_weights(jj).weights{2})));
      conv_layer_ids(ii) = jj;
    end
  end
end

%% tranplant paramters into full-convolutional net
trans_params = struct('weights', cell(numel(conv_names), 1), ...
  'layer_names', cell(numel(conv_names), 1));
for ii = 1:numel(conv_names)
  trans_params(ii).layer_names = conv_names{ii};
  weights = cell(2, 1);
  weights{1} = reshape(fc_weights(fc_layer_ids(ii)).weights{1}, size(conv_weights(conv_layer_ids(ii)).weights{1}));
  weights{2} = reshape(fc_weights(fc_layer_ids(ii)).weights{2}, size(conv_weights(conv_layer_ids(ii)).weights{2}));
  trans_params(ii).weights = weights;
end
caffe('set_weights', trans_params);
%% save
fully_conv_model_file = fullfile(caffe_root, 'examples/imagenet/bvlc_caffenet_full_conv.caffemodel');
caffe('save', fully_conv_model_file);

