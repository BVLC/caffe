addpath('/rmt/work/deeplabel/code/matlab/my_script');
SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for w = range_bi_w         %[3 5 7 9 11]                %0.5:0.5:6  %[1 5 10 15 20]
  bi_w = w;
  for x_std = range_bi_x_std   %35:5:65               % [10 20 30 40 50]
    bi_x_std = x_std;
    for r_std = range_bi_r_std   %5:5:10    % 5:5:20              % [10 20 30 40 50]
      bi_r_std = r_std;

      for pw = range_pos_w
        pos_w = pw;

        for p_x_std = range_pos_x_std
          pos_x_std = p_x_std;

          if down_sample_method == 1
	      if learn_crf
                  post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_ModelType%d_Epoch%d_downsampleBy%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, model_type, epoch, down_sample_rate);
              else 
                  post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_downsampleBy%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, down_sample_rate);
              end
          elseif down_sample_method == 2
	      if learn_crf
  	        post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_ModelType%d_Epoch%d_numSample%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, model_type, epoch, num_sample);
              else
                post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_numSample%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, num_sample);
              end
          else
            error('Wrong down_sample_method')
          end

          if is_server
            if learn_crf
              map_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'densecrf', 'res', feature_name, model_name, testset, feature_type, post_folder); 

              save_root_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, feature_type, post_folder); ;
            else
              map_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, model_name, testset, feature_type, post_folder);
              save_root_folder = map_folder;
            end
          else 
            map_folder = '../result';
          end

          map_dir = dir(fullfile(map_folder, '*.bin'));

          

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          if strcmp(dataset, 'voc12')
              seg_res_dir = [save_root_folder '/results/VOC2012/'];
          elseif strcmp(dataset, 'coco')
              seg_res_dir = [save_root_folder '/results/COCO2014/'];
          else
              error('Wrong dataset!')
          end

          save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

          if ~exist(save_result_folder, 'dir')
            mkdir(save_result_folder);
          end
  
          for i = 1 : numel(map_dir)
            fprintf(1, 'processing %d (%d)...\n', i, numel(map_dir));
            map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

            img_fn = map_dir(i).name(1:end-4);
            imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
          end

        end
      end
    end
  end
end
