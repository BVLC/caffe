'use strict';

var fs         = require('fs-extra');
var existsSync = require('exists-sync');

function touch(path, obj) {
  if (!existsSync(path)) {
    fs.createFileSync(path);
    fs.writeJsonSync(path, obj || {});
  }
}

function replaceFile(path, findString, replaceString) {
  if (existsSync(path)) {
    var newFile;
    var file = fs.readFileSync(path, 'utf-8');
    var find = new RegExp(findString);
    var match = new RegExp(replaceString);
    if (!file.match(match)) {
      newFile = file.replace(find, replaceString);
      fs.writeFileSync(path, newFile, 'utf-8');
    }
  }
}

module.exports = {
  touch:       touch,
  replaceFile: replaceFile
};
