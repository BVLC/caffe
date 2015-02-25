function res=read_cell(filename,linesep,cellsep)
if nargin < 2, linesep='\n'; end
if nargin < 3, cellsep = '\t'; end
if exist(filename,'file')
    fid = fopen(filename);
else
    % Assume that filename is either a file ide or a string
    fid = filename;
end

fileLines = textscan(fid,'%s','delimiter',linesep,'BufSize',100000);

fileLines = fileLines{1};

if regexp(fileLines{1},cellsep,'once')
    fileLines = regexprep(fileLines,['^' cellsep '|' cellsep '$'],'');
    res = regexp(fileLines,cellsep,'split');
    res = cell2matcell(res);
else
    res = fileLines;
end
