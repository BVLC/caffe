var test = require('tape');
var shimmedKeys = require('./index.js');
var is = require('is');
var keys = require('./shim.js');
var forEach = require('foreach');
var indexOf = require('indexof');

test('works', function (t) {
	var obj = {
		"str": "boz",
		"obj": {},
		"arr": [],
		"bool": true,
		"num": 42,
		"aNull": null,
		"undef": undefined
	};
	var objKeys = ['str', 'obj', 'arr', 'bool', 'num', 'aNull', 'undef'];

	t.test('exports a function', function (st) {
		if (Object.keys) {
			st.equal(Object.keys, shimmedKeys, 'Object.keys is supported and exported');
		} else {
			st.equal(keys, shimmedKeys, 'Object.keys is not supported; shim is exported');
		}
		st.end();
	});

	t.test('working with actual shim', function (st) {
		st.notEqual(Object.keys, keys, 'keys shim is not native Object.keys');
		st.end();
	});

	t.test('works with an object literal', function (st) {
		var theKeys = keys(obj);
		st.equal(is.array(theKeys), true, 'returns an array');
		st.deepEqual(theKeys, objKeys, 'Object has expected keys');
		st.end();
	});

	t.test('works with an array', function (st) {
		var arr = [1, 2, 3];
		var theKeys = keys(arr);
		st.equal(is.array(theKeys), true, 'returns an array');
		st.deepEqual(theKeys, ['0', '1', '2'], 'Array has expected keys');
		st.end();
	});

	t.test('returns names which are own properties', function (st) {
		forEach(keys(obj), function (name) {
			st.equal(obj.hasOwnProperty(name), true, name + ' should be returned');
		});
		st.end();
	});

	t.test('returns names which are enumerable', function (st) {
		var k, loopedValues = [];
		for (k in obj) {
			loopedValues.push(k);
		}
		forEach(keys(obj), function (name) {
			st.notEqual(indexOf(loopedValues, name), -1, name + ' is not enumerable');
		});
		st.end();
	});

	t.test('throws an error for a non-object', function (st) {
		st.throws(
			function () { return keys(42); },
			new TypeError('Object.keys called on a non-object'),
			'throws on a non-object'
		);
		st.end();
	});
	t.end();
});

test('works with an object instance', function (t) {
	var Prototype = function () {};
	Prototype.prototype.foo = true;
	var obj = new Prototype();
	obj.bar = true;
	var theKeys = keys(obj);
	t.equal(is.array(theKeys), true, 'returns an array');
	t.deepEqual(theKeys, ['bar'], 'Instance has expected keys');
	t.end();
});

