'use strict';
var path = require('path');
var osHomedir = require('os-homedir');
var home = osHomedir();

module.exports = function (str) {
	str = path.normalize(str) + path.sep;
	return str.replace(home + path.sep, '~' + path.sep).slice(0, -1);
};
