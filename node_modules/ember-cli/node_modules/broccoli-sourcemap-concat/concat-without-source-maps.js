'use strict';
var fs = require('fs');
var path = require('path');

module.exports = Concat;
function Concat(attrs) {
  this._internal = '';
  this.outputFile = attrs.outputFile;
  this.baseDir = attrs.baseDir;
}

Concat.prototype.addFile = function(file) {
  this._internal += fs.readFileSync(path.join(this.baseDir, file), 'UTF-8');
};

Concat.prototype.addSpace = function(space) {
  this._internal += space;
};

Concat.prototype.end = function() {
  fs.writeFileSync(this.outputFile, this._internal);
};
