module.exports = function canUseInputFiles(inputFiles) {
  if (!Array.isArray(inputFiles)) { return false; }

  return inputFiles.filter(function(file) {
    return !/[\{\}\|\*\?\!]/.test(file);
  }).length === inputFiles.length;
};
