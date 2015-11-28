'use strict';
var stripAnsi = require('strip-ansi');

module.exports = function (str) {
	var reAstral = /[\uD800-\uDBFF][\uDC00-\uDFFF]/g;

	return stripAnsi(str).replace(reAstral, ' ').length;
};
