'use strict';

module.exports = BuildError;

function BuildError(input){
  Error.call(this);
  this.message = input.message;
  this.file = input.file;
  this.filename = input.filename; // For testing errors from Uglify
  this.line = input.line;
  this.col = input.col;
  this.stack = input.stack;
}

BuildError.prototype = Object.create(Error.prototype);
