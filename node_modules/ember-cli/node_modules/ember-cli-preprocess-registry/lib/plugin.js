'use strict';

var path       = require('path');
var existsSync = require('exists-sync');

function Plugin(name, ext, options) {
  this.name = name;
  this.ext = ext;
  this.options = options || {};
  this.registry = this.options.registry;
  this.applicationName = this.options.applicationName;

  if (this.options.toTree) {
    this.toTree = this.options.toTree;
  }
}

Plugin.prototype.toTree = function() {
  throw new Error('A Plugin must implement the `toTree` method.');
};

Plugin.prototype.getExt = function(inputTreeRoot, inputPath, filename) {
  if(Array.isArray(this.ext)) {
    var detect = require('lodash/collection/find');
    return detect(this.ext, function(ext) {
      var filenameAndExt = filename + '.' + ext;
      return existsSync(path.join(inputTreeRoot, inputPath, filenameAndExt));
    });
  } else {
    return this.ext;
  }
};

module.exports = Plugin;
