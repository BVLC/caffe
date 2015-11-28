'use strict';

var findUp = require('findup-sync');
var path = require('path');

module.exports = function(file) {
  var buildFilePath = findUp(file);

  // Note
  // In the future this should throw
  if (buildFilePath === null) {
    return null;
  }

  var baseDir = path.dirname(buildFilePath);

  process.chdir(baseDir);

  var buildFile = require(buildFilePath);

  return buildFile;
};
