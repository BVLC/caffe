function varargout = matcaffe_getscores(im, varargin)
% [scores, maxlabel] = matcaffe_getscores(im, varargin)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% INPUTS
% im: color image as uint8 HxWx3
% 
% PARAMETERS
% b_force_init: force initialization of model (e.g., if using a new model definition)
%  (default: false)
% prediction_type: one of 'imagenet' (default) or 'features' for feature extractor
% b_print_results: print top image matches (only valid if prediction_type = 'imagenet')
%  (default: true)
%
% OUTPUTS
% scores: 1000-dimensional ILSVRC score vector
% maxlabel: label index with the highest score
%
% Usage:
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
%
% If you have multiple images, cat them with cat(4, ...)

% By Daniel Golden (dan at cellscope dot com) August 2014

%% Parse input parameters
p = inputParser;
p.addParameter('b_force_init', false);
p.addParameter('prediction_type', 'imagenet');
p.addParameter('b_print_results', true);
p.parse(varargin{:});

%% Go

init(p.Results.prediction_type, p.Results.b_force_init);

if nargin < 1
  % For demo purposes we will use the peppers image
  im = imread('peppers.png');
end

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image_ilsvrc(im)};
toc;

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;

scores = scores{1};
scores = squeeze(scores);
scores = mean(scores,2);

[~,maxlabel] = max(scores);

if strcmp(p.Results.prediction_type, 'imagenet') && p.Results.b_print_results
  % If we're doing imagenet classifications, show the top results
  score_table = print_top_scores(scores, 10);
  plot_top_scores(score_table, 10);
end

if nargout > 0
  varargout{1} = scores;
end
if nargout > 1
  varargout{2} = maxlabel;
end

function score_table = print_top_scores(score, num_to_print)
%% Print the top N scores

[label, label_idx] = get_imagenet_labels;

score_table = table(label, label_idx, score);
score_table_sorted = sortrows(score_table, 'score', 'descend');

fprintf('Top Scores:\n');

disp(score_table_sorted(1:num_to_print, :));

function plot_top_scores(score_table, num_to_print)
%% Plot a bar graph of the top N scores

score_table_sorted = sortrows(score_table, 'score', 'descend');

figure;
figure_grow(gcf, 2, 1);
barh(1:num_to_print, score_table_sorted.score(1:num_to_print));
set(gca, 'ytick', 1:num_to_print, 'yticklabel', score_table_sorted.label(1:num_to_print));
xlabel('Score');

function images = prepare_image_ilsvrc(im)
%% Prepare ilsvrc image

d = load('ilsvrc_2012_mean');
images = matcaffe_prepare_image(im, d.image_mean);

function init(prediction_type, b_force_init)
%% Init network

persistent last_prediction_type

switch prediction_type
  case 'imagenet'
    model_def_filename = fullfile(getenv('CAFFE_HOME'), 'examples/imagenet/imagenet_deploy.prototxt');
  case 'features'
    model_def_filename = fullfile(getenv('CAFFE_HOME'), 'matlab/caffe/imagenet_feature_extractor.prototxt');
  otherwise
    error('Invalid prediction_type: %s', prediction_type);
end

if ~isequal(prediction_type, last_prediction_type)
  % Prediction type has changed; need to force init caffe
  b_force_init = true;
end

matcaffe_init('model_def_file', model_def_filename, 'b_force', b_force_init);

last_prediction_type = prediction_type;
