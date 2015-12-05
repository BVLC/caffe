'use strict';

var path            = require('path');
var nodeModulesPath = require('node-modules-path');

module.exports = function requireLocal(lib) {
  return require(path.join(nodeModulesPath(process.cwd()), lib));
};
