'use strict';
var crypto = require('crypto');

module.exports = function (buf) {
	return crypto.createHash('md5').update(buf, 'utf8').digest('hex');
};
