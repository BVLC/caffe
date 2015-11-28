'use strict';
var packageJson = require('package-json');

module.exports = function (name, cb) {
	packageJson(name.toLowerCase(), 'latest', function (err, json) {
		if (err) {
			cb(err);
			return;
		}

		cb(null, json.version);
	});
};
