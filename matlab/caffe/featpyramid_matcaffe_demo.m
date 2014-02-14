
% Demo of the matlab wrapper to construct ConvNet feature pyramids
%
% input
%   imfn     image filename
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   N row structure array with two fields: scale, feats
%	 scale: scalar
%        feats: 2D numeric array
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  imfn = '../../examples/cat.jpg';
%  pyra = featpyramid_matcaffe_demo(imfn, 1);

function pyra = featpyramid_matcaffe_demo(imfn, use_gpu)
    if ~exist('imfn', 'var') 
        imfn = './pascal_000001.jpg'
        %imfn = '../../python/caffe/imagenet/pascal_009959.jpg';
    end

    %model_def_file = '../../examples/imagenet_deploy.prototxt';
    model_def_file = '../../python/caffe/imagenet/imagenet_rcnn_batch_1_input_1100x1100_output_conv5.prototxt' 
    % NOTE: you'll have to get the pre-trained ILSVRC network
    model_file = '../../examples/alexnet_train_iter_470000';

    %is_init = caffe('is_init');
    %display(['    (before calling init) caffe is init? = ' num2str(is_init)]);

    % init caffe network (spews logging info)
    caffe('init', model_def_file, model_file);

    %is_init = caffe('is_init');
    %display(['    (after calling init) caffe is init? = ' num2str(is_init)]);

    % set to use GPU or CPU
    if exist('use_gpu', 'var') && use_gpu
      caffe('set_mode_gpu');
    else
      caffe('set_mode_cpu');
    end

    % put into test mode
    caffe('set_phase_test');

    % optionally, pass parmeters as second argument to convnet_featpyramid (set here to the defaults)
    pyra_params.interval = 10;
    pyra_params.img_padding = 16;
    pyra_params.feat_minWidth = 1;
    pyra_params.feat_minHeight = 1;

    pyra = caffe('convnet_featpyramid', imfn, pyra_params ); % call with parameters ...
    % pyra = caffe('convnet_featpyramid', imfn ); % ... or with no parameters

    %keyboard
    %visualize one scale:
    colormap(gray);
    imagesc(squeeze(sum(pyra.feat{1}, 1)));


