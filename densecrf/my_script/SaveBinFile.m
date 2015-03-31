function SaveBinFile(data, fn, type)
% save as binary file

fid = fopen(fn, 'wb');

row = size(data, 1);
col = size(data, 2);
channel = size(data, 3);

fwrite(fid, row, 'int32');
fwrite(fid, col, 'int32');
fwrite(fid, channel, 'int32');

if strcmp(type, 'double')
    fwrite(fid, data(:), 'double');
elseif strcmp(type, 'single') || strcmp(type, 'float')
    fwrite(fid, data(:), 'single');
elseif strcmp(type, 'uint8')
    fwrite(fid, data(:), 'uint8');
elseif strcmp(type, 'int32')
    fwrite(fid, data(:), 'int32');    
else
    error('wrong type')
end

fclose(fid);