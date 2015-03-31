function out = LoadBinFile(fn, type)
% load binary file

fid = fopen(fn, 'rb');

row = fread(fid, 1, 'int32');
col = fread(fid, 1, 'int32');
channel = fread(fid, 1, 'int32');

numel = row * col * channel;

if strcmp(type, 'double')
    out = fread(fid, numel, 'double');
    out = double(out);
elseif strcmp(type, 'single') || strcmp(type, 'float')
    out = fread(fid, numel, 'single');   
    out = single(out);
elseif strcmp(type, 'uint8')
    out = fread(fid, numel, 'uint8');
    out = uint8(out);
elseif strcmp(type, 'int16')
    out = fread(fid, numel, 'int16');
    out = int16(out);
elseif strcmp(type, 'int32')
    out = fread(fid, numel, 'int32');
    out = int32(out);
else
    error('wrong type')
end

out = reshape(out, [row, col, channel]);

fclose(fid);