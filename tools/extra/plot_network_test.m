%plot the caffe neural network testing result
close all;
clear;
clc;
set(0,'DefaultTextInterpreter','none');

% load the log file of caffe model  
fid = fopen('C:\Projects\caffe\examples\cifar10_xnor\log\cifar10xnor_test.log', 'r');
tline = fgetl(fid);  


accuracyArray =[];
lossArray = [];
str = '';
str2 = '';
Net_name = 'CIFAR10';
%Net_name = 'CIFAR10_Binary';
%Net_name = 'CIFAR10_XNOR';

%record the last line  
lastLine = '';  
      
%read line  
while ischar(tline)
%%%%%%%%%%%%%% the accuracy line %%%%%%%%%%%%%%  
    k = strfind(tline, 'accuracy = ');
    k1 = strfind(tline, 'loss = ');
    k2 = strfind(tline, 'Batch');
    
    if(~isempty(k2))
        if(~isempty(k))
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k + 11;   
            indexEnd = size(tline);  
            str = tline(indexStart : indexEnd(2));  
            accuracyArray = [accuracyArray, str2double(str)];
        elseif(~isempty(k1))
            % If the string contain test and accuracy at the same time  
            % The bias from 'accuracy' to the float number  
            indexStart = k1 + 7;   
            indexEnd = size(tline);  
            str2 = tline(indexStart : indexEnd(2));  
            lossArray = [lossArray, str2double(str2)];
        else
            
        end
    else
        if(~isempty(k))
            indexStart = k + 11;   
            indexEnd = size(tline);  
            str = tline(indexStart : indexEnd(2));  
            avg_accuracy = str2double(str);
        elseif(~isempty(k1))
			
			k3 = strfind(tline, ' (* ');
			
            indexStart = k1 + 7;   
            indexEnd = k3;
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
h1 = plot(0:(length(accuracyArray)-1), accuracyArray);title([Net_name ': Accuracy vs Batch']);
lgd = legend('Accuracy');
title(lgd,['Avg Accuracy = ' num2str(avg_accuracy)]);
subplot(3,1,2);
h2 = plot(0:(length(lossArray)-1), lossArray);title([Net_name ': Loss vs Batch']);
lgd = legend('Loss');
title(lgd,['Avg Loss = ' num2str(avg_loss)]);
subplot(3,1,3);
h3 = plot(lossArray, accuracyArray,'*');title([Net_name ': Accuracy vs Loss']);

saveas(h,[Net_name '_test_result.jpg']);
