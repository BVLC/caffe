'use strict';
var onExit = require('signal-exit');
var api = require('./api');
var installed = false;

function outputRejectedMessage(err) {
	if (err instanceof Error) {
		console.error(err.stack);
	} else if (typeof err === 'undefined') {
		console.error('Promise rejected no value');
	} else {
		console.error('Promise rejected with value:', err);
	}
}

module.exports = function () {
	if (installed) {
		console.trace('WARN: loud rejection called more than once');
		return;
	}

	installed = true;

	var tracker = api(process);

	onExit(function () {
		var unhandledRejections = tracker.currentlyUnhandled();

		if (unhandledRejections.length > 0) {
			unhandledRejections.forEach(function (x) {
				outputRejectedMessage(x.reason);
			});

			process.exitCode = 1;
		}
	});
};
