'use strict';

var ioUtils = require('./utils/io-utils');
var path    = require('path');

var readFile    = ioUtils.readFile;
var isFile      = ioUtils.isFile;
var isDirectory = ioUtils.isDirectory;

function Config(localPath) {
  var content = {};

  if (!localPath) { return content; }

  localPath = path.normalize(localPath);

  if (isDirectory(localPath)) {
    this.outputError(localPath);
    return content;
  } else if (isFile(localPath)) {
    return readFile(localPath);
  }

  return content;
}

Config.prototype.outputError = function(localPath) {
  console.error('Settings file ' + localPath + ' should be a file, not a directory.');
};

module.exports = Config;
