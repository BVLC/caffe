'use strict';

module.exports = function getPathOption(options) {
  var outputPath     = 'components';
  if (options.path) {
    outputPath = options.path;
  } else {
    if (options.path === '') {
      outputPath = '';
    }
  }
  return outputPath;
};
