'use strict';

var fsExtra = require('fs-extra');

var isDirectory = function(path) {
  var stat = exists(path);

  if (!stat) { return false; }

  return stat.isDirectory();
};

var isFile = function(path) {
  var stat = exists(path);

  if (!stat) { return false; }

  return stat.isFile();
};

var exists = function(path) {
  var stat;

  try {
    stat = fsExtra.statSync(path);
  } catch(e) {
    return false;
  }

  return stat;
};

var stripComments = function stripComments(string) {
  string = string || "";

  string = string.replace(/\/\*(?:(?!\*\/)[\s\S])*\*\//g, "");
  string = string.replace(/\/\/\s\S.+/g, ""); // Everything after '//'

  return string;
};

var readFile = function readFile(path) {
  var contents = fsExtra.readFileSync(path, { encoding: 'utf8' });
  contents = stripComments(contents);

  if(!contents.length) {
    return {};
  }

  try {
    return JSON.parse(contents);
  } catch(e) {
    throw "Error when parsing file in " + path + ". Make sure that you have a valid JSON."
  }
};

module.exports.readFile    = readFile;
module.exports.isFile      = isFile;
module.exports.isDirectory = isDirectory;
