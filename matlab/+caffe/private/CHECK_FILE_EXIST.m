function CHECK_FILE_EXIST(filename)

if exist(filename, 'file') == 0
  error('%s does not exist', filename);
end

end
