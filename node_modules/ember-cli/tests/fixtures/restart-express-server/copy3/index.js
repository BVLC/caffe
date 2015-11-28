'use strict';

var fs = require('fs');
var path = require('path');
var aCache = require.cache[path.resolve('./subfolder/a')];
var bCache = require.cache[path.resolve('./subfolder/b')];

module.exports = function () {
  fs.writeFileSync('foo.txt', !aCache + ' ' + !bCache);
};
