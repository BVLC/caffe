'use strict';

var path = require('path');

function getUserHomeDirectory() {
  return process.env[(process.platform == 'win32') ? 'USERPROFILE' : 'HOME'];
}

function getCurrentPath() {
  return process.cwd();
}

module.exports = {
  getUserHomeDirectory: getUserHomeDirectory,
  getCurrentPath:       getCurrentPath
};
