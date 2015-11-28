'use strict';

var path    = require('path');
var Promise = require('../../lib/ext/promise');
var copy    = Promise.denodeify(require('cpr'));

var rootPath = process.cwd();

module.exports = function copyFixtureFiles(sourceDir) {
  return copy(path.join(rootPath, 'tests', 'fixtures', sourceDir), '.');
};
