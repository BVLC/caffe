%plot the caffe neural network testing result
close all;
clear;
clc;
set(0,'DefaultTextInterpreter','none');

% load the log file of caffe model
file_path = 'H:\Caffe_Log\log\bvlc_reference_alexnet_inq_test.log';
fid = fopen(file_path, 'r');
[filepath,name,ext] = fileparts(file_path);
tline = fgetl(fid);  

accuracy_top1_Array =[];
accuracy_top5_Array =[];
lossArray = [];

str = '';
str2 = '';
Net_name = name;

%record the last line
lastLine = '';  

%Seek the words from log
str_index_acc1 = 'accuracy1 = ';
str_index_acc5 = 'accuracy5 = ';
str_index_loss = 'loss = ';
str_index_batch = 'Batch';
str_index_symbol = ' (* ';

%read line  
while ischar(tline)
%%%%%%%%%%%%%% the accuracy line %%%%%%%%%%%%%%  
    k = strfind(tline, str_index_acc1);
	k1 = strfind(tline, str_index_acc5);
    k2 = strfind(tline, str_index_loss);
    k3 = strfind(tline, str_index_batch);
	k4 = strfind(tline, str_index_symbol);
    
    if(~isempty(k3))
        if(~isempty(k))
            indexStart = k + length(str_index_acc1);   
            indexEnd = size(tline);  
            str = tline(indexStart : indexEnd(2));  
            accuracy_top1_Array = [accuracy_top1_Array, str2double(str)];
        elseif(~isempty(k1))
            indexStart = k1 + length(str_index_acc5);   
            indexEnd = size(tline);  
            str2 = tline(indexStart : indexEnd(2));  
            accuracy_top5_Array = [accuracy_top5_Array, str2double(str2)];
		elseif(~isempty(k2))
            indexStart = k2 + length(str_index_loss);   
            indexEnd = size(tline);  
            str2 = tline(indexStart : indexEnd(2));  
            lossArray = [lossArray, str2double(str2)];
		else
        end
    else
        if(~isempty(k))
            indexStart = k + length(str_index_acc1);   
            indexEnd = size(tline);  
            str = tline(indexStart : indexEnd(2));  
            avg_accuracy_top1 = str2double(str);
		elseif(~isempty(k1))
            indexStart = k1 + length(str_index_acc5);   
            indexEnd = size(tline);  
            str = tline(indexStart : indexEnd(2));  
            avg_accuracy_top5 = str2double(str);
        elseif(~isempty(k2))
            indexStart = k2 + length(str_index_loss);   
            indexEnd = k4;
            str = tline(indexStart : indexEnd);  
            avg_loss = str2double(str);
        else
            
        end
    end

    lastLine = tline;  
    tline = fgetl(fid);      
end  

%draw figure
h = figure('Position', [100, 100, 800, 800]);

subplot(3,1,1);
h1 = plot(0:(length(accuracy_top1_Array)-1), accuracy_top1_Array);title([Net_name ': Top 1 Accuracy vs Batch']);
lgd = legend('Top 1 Accuracy');
title(lgd,['Avg Accuracy = ' num2str(avg_accuracy_top1)]);

subplot(3,1,2);
h1 = plot(0:(length(accuracy_top5_Array)-1), accuracy_top5_Array);title([Net_name ': Top 5 Accuracy vs Batch']);
lgd = legend('Top 5 Accuracy');
title(lgd,['Avg Accuracy = ' num2str(avg_accuracy_top5)]);

subplot(3,1,3);
h2 = plot(0:(length(lossArray)-1), lossArray);title([Net_name ': Loss vs Batch']);
lgd = legend(['Avg Loss = ' num2str(avg_loss)]);

%save the figure as jpg
saveas(h, fullfile(filepath, [Net_name '_result']), 'jpeg');
