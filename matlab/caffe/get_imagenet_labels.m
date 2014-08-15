function [label, label_idx] = get_imagenet_labels
% Load imagenet label
% [label, label_idx] = get_imagenet_labels

% By Daniel Golden (dan at cellscope dot com)

%% Find file
label_filename = fullfile(getenv('CAFFE_HOME'), 'data', 'ilsvrc12', 'synset_words.txt');
assert(exist(label_filename, 'file') ~= 0, 'Label file does not exist: %s', label_filename);

%% Load
labels_trimmed = strtrim(fileread(label_filename));
label = strsplit(labels_trimmed, sprintf('\n'))';
label = regexprep(label, '^n\d+ ', '');

label_idx = (1:length(label))';

