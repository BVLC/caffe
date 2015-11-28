/*!
 * output-file-sync | MIT (c) Shinnosuke Watanabe
 * https://github.com/shinnn/output-file-sync
*/
'use strict';

var dirname = require('path').dirname;
var writeFileSync = require('fs').writeFileSync;
var xtend = require('xtend');

var mkdirpSync = require('mkdirp').sync;

module.exports = function outputFileSync(filePath, data, options) {
  options = options || {};

  var mkdirpOptions;
  if (typeof options === 'string') {
    mkdirpOptions = null;
  } else {
    if (options.dirMode) {
      mkdirpOptions = xtend(options, {mode: options.dirMode});
    } else {
      mkdirpOptions = options;
    }
  }

  var writeFileOptions;
  if (options.fileMode) {
    writeFileOptions = xtend(options, {mode: options.fileMode});
  } else {
    writeFileOptions = options;
  }

  var createdDirPath = mkdirpSync(dirname(filePath), mkdirpOptions);
  writeFileSync(filePath, data, writeFileOptions);
  return createdDirPath;
};
