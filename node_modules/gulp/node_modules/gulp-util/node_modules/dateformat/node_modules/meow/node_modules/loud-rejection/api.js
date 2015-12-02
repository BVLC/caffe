'use strict';

// WARNING: This undocumented API is subject to change.

module.exports = function (process) {
	var unhandledRejections = [];

	process.on('unhandledRejection', function (reason, p) {
		unhandledRejections.push({reason: reason, promise: p});
	});

	process.on('rejectionHandled', function (p) {
		var index = unhandledRejections.reduce(function (result, item, idx) {
			return (item.promise === p ? idx : result);
		}, -1);

		unhandledRejections.splice(index, 1);
	});

	function currentlyUnhandled() {
		return unhandledRejections.map(function (entry) {
			return {
				reason: entry.reason,
				promise: entry.promise
			};
		});
	}

	return {
		currentlyUnhandled: currentlyUnhandled
	};
};
