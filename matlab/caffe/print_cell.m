function res=print_cell(input,file,linesep,cellsep)
assert(iscell(input),'The input should be a cell')
if nargin < 4
    cellsep = '\t';
end
if nargin < 3
    linesep = '\n';
end
if exist('file','var') && ~isempty(file)
    %%
    fid = fopen(file,'w');
    for l=1:length(input)
        if iscell(input{l})
            for i=1:length(input{l})
                fprintf(fid,['%s' cellsep],input{l}{i});
            end
            fprintf(fid,linesep);
        else
            if size(input,2) > 1
                for i=1:size(input,2)
                    fprintf(fid,'%s ',input{l,i});
                end
                fprintf(fid,linesep);
            else
                fprintf(fid,['%s' linesep],input{l});
            end
        end
    end
    fclose(fid);
else
    res = '';
    for l=1:length(input)
        if iscell(input{l})
            for i=1:length(input{l})
                res = [res sprintf([cellsep{1} '%s' cellsep{2}],input{l}{i})];
            end
            res = [res sprintf(linesep)];
        else
            res = [res sprintf(['%s' linesep],input{l}(:))];
        end
    end
end