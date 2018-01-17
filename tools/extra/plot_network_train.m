%plot the caffe neural network training result
close all;
clc;
clear;
set(0,'DefaultTextInterpreter','none');

%# Carry out testing every 1000 training iterations.
%test_interval: 1000
%# Display every 200 iterations
%display: 200

% load the log file of caffe model
fid = fopen('C:\Projects\caffe\examples\cifar10_binary\log\cifar10binary_train.log', 'r');
tline = fgetl(fid);  

accuracyArray =[];
lossArray = [];

accuracyStep = 1000;
lossStep = 200;

str = '';
str2 = '';

Net_name = 'CIFAR10_Binary';

%record the last line  
lastLine = '';  
      
%read line  
while ischar(tline)
	
	%%%%%%%%%%%%%% the accuracy line %%%%%%%%%%%%%%  
	k = strfind(tline, 'Test net output');
	k1 = strfind(tline, 'Iteration');
	indexEnd = size(tline);
	
	if(~isempty(k))
		k2 = strfind(tline, 'accuracy = ');  
		if(~isempty(k2))
			indexStart = k2 + 11;
			str = tline(indexStart : indexEnd(2));
			accuracyArray = [accuracyArray, str2double(str)];
		end
		
	elseif(~isempty(k1))
		%%%%%%%%%%%%%% the loss line %%%%%%%%%%%%%%
		k2 = strfind(tline, 'loss = ');
		if(~isempty(k2))
			indexStart = k2 + 7;
			str2 = tline(indexStart : indexEnd(2));
			lossArray = [lossArray, str2double(str2)];
		end
	end
	
	lastLine = tline;
	tline = fgetl(fid);
end

%draw figure
h = figure('Position', [100, 100, 800, 800]);
subplot(2,1,1);
h1 = plot(0:accuracyStep:(length(accuracyArray)-1)*accuracyStep, accuracyArray);title([Net_name ': iteration vs accuracy']);
subplot(2,1,2);
h2 = plot(0:lossStep:(length(lossArray)-1)*lossStep, lossArray);title([Net_name ': iteration vs loss']);
saveas(h,[Net_name '_train_result.jpg']);
