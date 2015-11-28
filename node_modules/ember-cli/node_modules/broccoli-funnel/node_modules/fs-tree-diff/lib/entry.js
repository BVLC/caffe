'use strict';

var DIRECTORY_MODE = 16877;

module.exports = Entry;
function Entry(relativePath, size, mtime) {
  var isDirectory = relativePath.charAt(relativePath.length - 1) === '/';
  this.relativePath = relativePath;
  this.mode = isDirectory ? DIRECTORY_MODE : 0;
  this.size = size;
  this.mtime = mtime;
}

Entry.prototype.isDirectory = function() {
  return (this.mode & 61440) === 16384;
};

