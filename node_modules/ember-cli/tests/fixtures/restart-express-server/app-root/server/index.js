'use strict';

var fs = require('fs');
var a = require('./subfolder/a');

module.exports = function () {
  fs.writeFileSync('foo.txt', a());
};
