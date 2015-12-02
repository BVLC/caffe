(function () {
	'use strict';

	var b64 = require('../lib/b64'),
		checks = [
			'a',
			'aa',
			'aaa',
			'hi',
			'hi!',
			'hi!!',
			'sup',
			'sup?',
			'sup?!'
		],
		res;

	res = checks.some(function (check) {
		var b64Str,
			arr,
			arr2,
			str,
			i,
			l;

		arr2 = [];
		for (i = 0, l = check.length; i < l; i += 1) {
			arr2.push(check.charCodeAt(i));
		}
		b64Str = b64.fromByteArray(arr2);

		arr = b64.toByteArray(b64Str);
		arr2 = [];
		for (i = 0, l = arr.length; i < l; i += 1) {
			arr2.push(String.fromCharCode(arr[i]));
		}
		str = arr2.join('');
		if (check !== str) {
			console.log('Fail:', check);
			console.log('Base64:', b64Str);
			return true;
		}
	});

	if (res) {
		console.log('Test failed');
	} else {
		console.log('All tests passed!');
	}
}());
