'use strict';
module.exports = function (val) {
	if (val == null) {
		return [];
	}

	return Array.isArray(val) ? val : [val];
};
