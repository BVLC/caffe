%plot the caffe neural network training result
close all;
clc;
clear;
set(0,'DefaultTextInterpreter','none');


% load the log file of caffe model  
file_path = 'H:\Caffe_Log\log\bvlc_alexnet_train_a.log';
fid = fopen(file_path, 'r');
[filepath,name,ext] = fileparts(file_path);
tline = fgetl(fid);  

accuracy_top1_Array =[];
accuracy_top5_Array =[];
accuracy_iteration_Array = [];
lossArray = [];
loss_iteration_Array = [];

accuracyStep = 1000;
lossStep = 200;

str = '';
str2 = '';

Net_name = name;

%record the last line  
lastLine = '';

%Seek the words from log
str_index_acc1 = 'accuracy_top_1 = ';
str_index_acc5 = 'accuracy_top_5 = ';
str_index_loss = 'loss = ';
str_index_netout = 'Test net output';
str_index_iteration = 'Iteration ';
str_index_net = ', Testing net';
      
%read line  
while ischar(tline)
	
	%%%%%%%%%%%%%% the accuracy line %%%%%%%%%%%%%%  
	k = strfind(tline, str_index_acc1);
	k1 = strfind(tline, str_index_acc5);
	k2 = strfind(tline, str_index_loss);
	k3 = strfind(tline, str_index_netout);
	k4 = strfind(tline, str_index_iteration);
	k5 = strfind(tline, str_index_net);
	indexEnd = size(tline);
	
	if(~isempty(k3))
		if(~isempty(k))
			indexStart = k + length(str_index_acc1);
			str = tline(indexStart : indexEnd(2));
			accuracy_top1_Array = [accuracy_top1_Array, str2double(str)];
		elseif(~isempty(k1))
			indexStart = k1 + length(str_index_acc1);
			str = tline(indexStart : indexEnd(2));
			accuracy_top5_Array = [accuracy_top5_Array, str2double(str)];
		else
		end
		
	elseif(~isempty(k4))
		if(~isempty(k5))
			indexStart = k4 + length(str_index_iteration);
			indexEnd = k5;
			str2 = tline(indexStart : indexEnd);
			accuracy_iteration_Array = [accuracy_iteration_Array, str2double(str2)];
			
		elseif(~isempty(k2))
			indexStart = k2 + length(str_index_loss);
			str2 = tline(indexStart : indexEnd(2));
			lossArray = [lossArray, str2double(str2)];
			
			indexStart = k4 + length(str_index_iteration);
			indexEnd = strfind(tline, ' (');
			str2 = tline(indexStart : indexEnd);
			loss_iteration_Array = [loss_iteration_Array, str2double(str2)];
		end
	end
	
	lastLine = tline;
	tline = fgetl(fid);
end

%draw figure
h = figure('Position', [100, 100, 800, 800]);
subplot(2,1,1);
plot(accuracy_iteration_Array, accuracy_top1_Array, 'r');
hold on;
plot(accuracy_iteration_Array, accuracy_top5_Array, 'b');
lgd = legend('Top 1 Accuracy', 'Top 5 Accuracy','Location','northwest');
hold off;
title([Net_name ': Accuracy vs Iteration']);

subplot(2,1,2);
plot(loss_iteration_Array, lossArray, 'k');
lgd = legend('Loss','Location','northwest');
title([Net_name ': Loss vs Iteration']);


%save the figure as jpg
saveas(h, fullfile(filepath, [Net_name '_result']), 'jpeg');
