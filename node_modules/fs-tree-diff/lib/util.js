'use strict';

function byRelativePath(entry) {
  return entry.relativePath;
}

function chomp(string, character) {
  if (string.charAt(string.length-1) === character) {
    return string.substring(0, string.length-1);
  } else {
    return string;
  }
}

module.exports.byRelativePath = byRelativePath;
module.exports.chomp = chomp;
