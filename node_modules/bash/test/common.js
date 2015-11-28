var common = exports;

exports.assert = require('assert');

common.require = function(lib) {
  return require('../lib/' + lib);
};
