'use strict';

var fs = require('fs');
var a = require('./subfolder/a');
var b = require('./subfolder/b');

module.exports = function () {
  fs.writeFileSync('foo.txt', a() + ' ' + b());
};
