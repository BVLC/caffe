var test = require('tape');
var is = require('../index.js');

var forEach = require('foreach');
var now = Date.now || function () { return +new Date(); };

test('is.type', function (t) {
  var booleans = [true, false];
  forEach(booleans, function (boolean) {
    t.ok(is.type(boolean, 'boolean'), '"' + boolean + '" is a boolean');
  });

  var numbers = [1, 0 / 1, 0 / -1, NaN, Infinity, -Infinity];
  forEach(numbers, function (number) {
    t.ok(is.type(number, 'number'), '"' + number + '" is a number');
  });

  var objects = [{}, null, new Date()];
  forEach(objects, function (object) {
    t.ok(is.type(object, 'object'), '"' + object + '" is an object');
  });

  var strings = ['', 'abc'];
  forEach(strings, function (string) {
    t.ok(is.type(string, 'string'), '"' + string + '" is a string');
  });

  t.ok(is.type(undefined, 'undefined'), 'undefined is undefined');

  t.end();
});

test('is.undefined', function (t) {
  t.ok(is.undefined(), 'undefined is undefined');
  t.notOk(is.undefined(null), 'null is not undefined');
  t.notOk(is.undefined({}), 'object is not undefined');
  t.end();
});

test('is.defined', function (t) {
  t.notOk(is.defined(), 'undefined is not defined');
  t.ok(is.defined(null), 'null is defined');
  t.ok(is.defined({}), 'object is undefined');
  t.end();
});

test('is.empty', function (t) {
  t.ok(is.empty(''), 'empty string is empty');
  t.ok(is.empty([]), 'empty array is empty');
  t.ok(is.empty({}), 'empty object is empty');
  (function () { t.ok(is.empty(arguments), 'empty arguments is empty'); }());
  t.end();
});

test('is.equal', function (t) {
  t.ok(is.equal([1, 2, 3], [1, 2, 3]), 'arrays are shallowly equal');
  t.ok(is.equal([1, 2, [3, 4]], [1, 2, [3, 4]]), 'arrays are deep equal');
  t.ok(is.equal({ a: 1, b: 2, c: 3 }, { a: 1, b: 2, c: 3 }), 'objects are shallowly equal');
  t.ok(is.equal({ a: { b: 1 } }, { a: { b: 1 } }), 'objects are deep equal');
  var nowTS = now();
  t.ok(is.equal(new Date(nowTS), new Date(nowTS)), 'two equal date objects are equal');

  var F = function () {};
  F.prototype = {};
  t.ok(is.equal(new F(), new F()), 'two object instances are equal when the prototype is the same');
  t.end();
});

test('is.hosted', function (t) {
  t.ok(is.hosted('a', { a: {} }), 'object is hosted');
  t.ok(is.hosted('a', { a: [] }), 'array is hosted');
  t.ok(is.hosted('a', { a: function () {} }), 'function is hosted');
  t.notOk(is.hosted('a', { a: true }), 'boolean value is not hosted');
  t.notOk(is.hosted('a', { a: false }), 'boolean value is not hosted');
  t.notOk(is.hosted('a', { a: 3 }), 'number value is not hosted');
  t.notOk(is.hosted('a', { a: undefined }), 'undefined value is not hosted');
  t.notOk(is.hosted('a', { a: 'abc' }), 'string value is not hosted');
  t.notOk(is.hosted('a', { a: null }), 'null value is not hosted');
  t.end();
});

test('is.instance', function (t) {
  t.ok(is.instance(new Date(), Date), 'new Date is instanceof Date');
  var F = function () {};
  t.ok(is.instance(new F(), F), 'new constructor is instanceof constructor');
  t.end();
});

test('is.null', function (t) {
  var isNull = is['null'];
  t.ok(isNull(null), 'null is null');
  t.notOk(isNull(undefined), 'undefined is not null');
  t.notOk(isNull({}), 'object is not null');
  t.end();
});

test('is.arguments', function (t) {
  t.notOk(is.arguments([]), 'array is not arguments');
  (function () { t.ok(is.arguments(arguments), 'arguments is arguments'); }());
  (function () { t.notOk(is.arguments(Array.prototype.slice.call(arguments)), 'sliced arguments is not arguments'); }());
  t.end();
});

test('is.array', function (t) {
  t.ok(is.array([]), 'array is array');
  (function () { t.ok(is.array(Array.prototype.slice.call(arguments)), 'sliced arguments is array'); }());
  t.end();
});

test('is.array.empty', function (t) {
  t.ok(is.array.empty([]), 'empty array is empty array');
  (function () { t.notOk(is.array.empty(arguments), 'empty arguments is not empty array'); }());
  (function () { t.ok(is.array.empty(Array.prototype.slice.call(arguments)), 'empty sliced arguments is empty array'); }());
  t.end();
});

test('is.arguments.empty', function (t) {
  t.notOk(is.arguments.empty([]), 'empty array is not empty arguments');
  (function () { t.ok(is.arguments.empty(arguments), 'empty arguments is empty arguments'); }());
  (function () { t.notOk(is.arguments.empty(Array.prototype.slice.call(arguments)), 'empty sliced arguments is not empty arguments'); }());
  t.end();
});

test('is.isarraylike', function (t) {
  t.notOk(is.arraylike(), 'undefined is not array-like');
  t.notOk(is.arraylike(null), 'null is not array-like');
  t.notOk(is.arraylike(false), 'false is not array-like');
  t.notOk(is.arraylike(true), 'true is not array-like');
  t.ok(is.arraylike({ length: 0 }), 'object with zero length is array-like');
  t.ok(is.arraylike({ length: 1 }), 'object with positive length is array-like');
  t.notOk(is.arraylike({ length: -1 }), 'object with negative length is not array-like');
  t.notOk(is.arraylike({ length: NaN }), 'object with NaN length is not array-like');
  t.notOk(is.arraylike({ length: 'foo' }), 'object with string length is not array-like');
  t.notOk(is.arraylike({ length: '' }), 'object with empty string length is not array-like');
  t.ok(is.arraylike([]), 'array is array-like');
  (function () { t.ok(is.arraylike(arguments), 'empty arguments is array-like'); }());
  (function () { t.ok(is.arraylike(arguments), 'nonempty arguments is array-like'); }(1, 2, 3));
  t.end();
});

test('is.boolean', function (t) {
  t.ok(is.boolean(true), 'literal true is a boolean');
  t.ok(is.boolean(false), 'literal false is a boolean');
  t.ok(is.boolean(new Boolean(true)), 'object true is a boolean');
  t.ok(is.boolean(new Boolean(false)), 'object false is a boolean');
  t.notOk(is.boolean(), 'undefined is not a boolean');
  t.notOk(is.boolean(null), 'null is not a boolean');
  t.end();
});

test('is.false', function (t) {
  var isFalse = is['false'];
  t.ok(isFalse(false), 'false is false');
  t.ok(isFalse(new Boolean(false)), 'object false is false');
  t.notOk(isFalse(true), 'true is not false');
  t.notOk(isFalse(), 'undefined is not false');
  t.notOk(isFalse(null), 'null is not false');
  t.notOk(isFalse(''), 'empty string is not false');
  t.end();
});

test('is.true', function (t) {
  var isTrue = is['true'];
  t.ok(isTrue(true), 'true is true');
  t.ok(isTrue(new Boolean(true)), 'object true is true');
  t.notOk(isTrue(false), 'false is not true');
  t.notOk(isTrue(), 'undefined is not true');
  t.notOk(isTrue(null), 'null is not true');
  t.notOk(isTrue(''), 'empty string is not true');
  t.end();
});

test('is.date', function (t) {
  t.ok(is.date(new Date()), 'new Date is date');
  t.notOk(is.date(), 'undefined is not date');
  t.notOk(is.date(null), 'null is not date');
  t.notOk(is.date(''), 'empty string is not date');
  t.notOk(is.date(now()), 'timestamp is not date');
  var F = function () {};
  F.prototype = new Date();
  t.notOk(is.date(new F()), 'Date subtype is not date');
  t.end();
});

test('is.element', function (t) {
  if (typeof HTMLElement !== 'undefined') {
    var element = document.createElement('div');
    t.ok(is.element(element), 'HTMLElement is element');
    t.notOk(is.element({ nodeType: 1 }), 'object with nodeType is not element');
  } else {
    t.ok(true, 'Skipping is.element test in a non-browser environment');
  }
  t.end();
});

test('is.error', function (t) {
  var err = new Error('foo');
  t.ok(is.error(err), 'Error is error');
  t.notOk(is.error({}), 'object is not error');
  t.notOk(is.error({ toString: function () { return '[object Error]'; } }), 'object with error\'s toString is not error');
  t.end();
});

test('is.fn', function (t) {
  t.equal(is['function'], is.fn, 'alias works');
  t.ok(is.fn(function () {}), 'function is function');
  t.ok(is.fn(console.log), 'console.log is function');
  if (typeof window !== 'undefined') {
    // in IE7/8, typeof alert === 'object'
    t.ok(is.fn(window.alert), 'window.alert is function');
  }
  t.notOk(is.fn({}), 'object is not function');
  t.notOk(is.fn(null), 'null is not function');
  t.end();
});

test('is.number', function (t) {
  t.ok(is.number(0), 'positive zero is number');
  t.ok(is.number(0 / -1), 'negative zero is number');
  t.ok(is.number(3), 'three is number');
  t.ok(is.number(NaN), 'NaN is number');
  t.ok(is.number(Infinity), 'infinity is number');
  t.ok(is.number(-Infinity), 'negative infinity is number');
  t.ok(is.number(new Number(42)), 'object number is number');
  t.notOk(is.number(), 'undefined is not number');
  t.notOk(is.number(null), 'null is not number');
  t.notOk(is.number(true), 'true is not number');
  t.end();
});

test('is.infinite', function (t) {
  t.ok(is.infinite(Infinity), 'positive infinity is infinite');
  t.ok(is.infinite(-Infinity), 'negative infinity is infinite');
  t.notOk(is.infinite(NaN), 'NaN is not infinite');
  t.notOk(is.infinite(0), 'a number is not infinite');
  t.end();
});

test('is.decimal', function (t) {
  t.ok(is.decimal(1.1), 'decimal is decimal');
  t.notOk(is.decimal(0), 'zero is not decimal');
  t.notOk(is.decimal(1), 'integer is not decimal');
  t.notOk(is.decimal(NaN), 'NaN is not decimal');
  t.notOk(is.decimal(Infinity), 'Infinity is not decimal');
  t.end();
});

test('is.divisibleBy', function (t) {
  t.ok(is.divisibleBy(4, 2), '4 is divisible by 2');
  t.ok(is.divisibleBy(4, 2), '4 is divisible by 2');
  t.ok(is.divisibleBy(0, 1), '0 is divisible by 1');
  t.ok(is.divisibleBy(Infinity, 1), 'infinity is divisible by anything');
  t.ok(is.divisibleBy(1, Infinity), 'anything is divisible by infinity');
  t.ok(is.divisibleBy(Infinity, Infinity), 'infinity is divisible by infinity');
  t.notOk(is.divisibleBy(1, 0), '1 is not divisible by 0');
  t.notOk(is.divisibleBy(NaN, 1), 'NaN is not divisible by 1');
  t.notOk(is.divisibleBy(1, NaN), '1 is not divisible by NaN');
  t.notOk(is.divisibleBy(NaN, NaN), 'NaN is not divisible by NaN');
  t.notOk(is.divisibleBy(1, 3), '1 is not divisible by 3');
  t.end();
});

test('is.int', function (t) {
  t.ok(is.int(0), '0 is integer');
  t.ok(is.int(3), '3 is integer');
  t.notOk(is.int(1.1), '1.1 is not integer');
  t.notOk(is.int(NaN), 'NaN is not integer');
  t.notOk(is.int(Infinity), 'infinity is not integer');
  t.notOk(is.int(null), 'null is not integer');
  t.notOk(is.int(), 'undefined is not integer');
  t.end();
});

test('is.maximum', function (t) {
  t.ok(is.maximum(3, [3, 2, 1]), '3 is maximum of [3,2,1]');
  t.ok(is.maximum(3, [1, 2, 3]), '3 is maximum of [1,2,3]');
  t.ok(is.maximum(4, [1, 2, 3]), '4 is maximum of [1,2,3]');
  t.ok(is.maximum('c', ['a', 'b', 'c']), 'c is maximum of [a,b,c]');
  t.notOk(is.maximum(2, [1, 2, 3]), '2 is not maximum of [1,2,3]');
  var error = new TypeError('second argument must be array-like');
  t.throws(function () { return is.maximum(2, null); }, error, 'throws when second value is not array-like');
  t.throws(function () { return is.maximum(2, {}); }, error, 'throws when second value is not array-like');
  t.end();
});

test('is.minimum', function (t) {
  t.ok(is.minimum(1, [1, 2, 3]), '1 is minimum of [1,2,3]');
  t.ok(is.minimum(0, [1, 2, 3]), '0 is minimum of [1,2,3]');
  t.ok(is.minimum('a', ['a', 'b', 'c']), 'a is minimum of [a,b,c]');
  t.notOk(is.minimum(2, [1, 2, 3]), '2 is not minimum of [1,2,3]');
  var error = new TypeError('second argument must be array-like');
  t.throws(function () { return is.minimum(2, null); }, error, 'throws when second value is not array-like');
  t.throws(function () { return is.minimum(2, {}); }, error, 'throws when second value is not array-like');
  t.end();
});

test('is.nan', function (t) {
  t.ok(is.nan(NaN), 'NaN is not a number');
  t.ok(is.nan('abc'), 'string is not a number');
  t.ok(is.nan(true), 'boolean is not a number');
  t.ok(is.nan({}), 'object is not a number');
  t.ok(is.nan([]), 'array is not a number');
  t.ok(is.nan(function () {}), 'function is not a number');
  t.notOk(is.nan(0), 'zero is a number');
  t.notOk(is.nan(3), 'three is a number');
  t.notOk(is.nan(1.1), '1.1 is a number');
  t.notOk(is.nan(Infinity), 'infinity is a number');
  t.end();
});

test('is.even', function (t) {
  t.ok(is.even(0), 'zero is even');
  t.ok(is.even(2), 'two is even');
  t.ok(is.even(Infinity), 'infinity is even');
  t.notOk(is.even(1), '1 is not even');
  t.notOk(is.even(), 'undefined is not even');
  t.notOk(is.even(null), 'null is not even');
  t.notOk(is.even(NaN), 'NaN is not even');
  t.end();
});

test('is.odd', function (t) {
  t.ok(is.odd(1), 'zero is odd');
  t.ok(is.odd(3), 'two is odd');
  t.ok(is.odd(Infinity), 'infinity is odd');
  t.notOk(is.odd(0), '0 is not odd');
  t.notOk(is.odd(2), '2 is not odd');
  t.notOk(is.odd(), 'undefined is not odd');
  t.notOk(is.odd(null), 'null is not odd');
  t.notOk(is.odd(NaN), 'NaN is not odd');
  t.end();
});

test('is.ge', function (t) {
  t.ok(is.ge(3, 2), '3 is greater than 2');
  t.notOk(is.ge(2, 3), '2 is not greater than 3');
  t.ok(is.ge(3, 3), '3 is greater than or equal to 3');
  t.ok(is.ge('abc', 'a'), 'abc is greater than a');
  t.ok(is.ge('abc', 'abc'), 'abc is greater than or equal to abc');
  t.notOk(is.ge('a', 'abc'), 'a is not greater than abc');
  t.notOk(is.ge(Infinity, 0), 'infinity is not greater than anything');
  t.notOk(is.ge(0, Infinity), 'anything is not greater than infinity');
  var error = new TypeError('NaN is not a valid value');
  t.throws(function () { return is.ge(NaN, 2); }, error, 'throws when first value is NaN');
  t.throws(function () { return is.ge(2, NaN); }, error, 'throws when second value is NaN');
  t.end();
});

test('is.gt', function (t) {
  t.ok(is.gt(3, 2), '3 is greater than 2');
  t.notOk(is.gt(2, 3), '2 is not greater than 3');
  t.notOk(is.gt(3, 3), '3 is not greater than 3');
  t.ok(is.gt('abc', 'a'), 'abc is greater than a');
  t.notOk(is.gt('abc', 'abc'), 'abc is not greater than abc');
  t.notOk(is.gt('a', 'abc'), 'a is not greater than abc');
  t.notOk(is.gt(Infinity, 0), 'infinity is not greater than anything');
  t.notOk(is.gt(0, Infinity), 'anything is not greater than infinity');
  var error = new TypeError('NaN is not a valid value');
  t.throws(function () { return is.gt(NaN, 2); }, error, 'throws when first value is NaN');
  t.throws(function () { return is.gt(2, NaN); }, error, 'throws when second value is NaN');
  t.end();
});

test('is.le', function (t) {
  t.ok(is.le(2, 3), '2 is lesser than or equal to 3');
  t.notOk(is.le(3, 2), '3 is not lesser than or equal to 2');
  t.ok(is.le(3, 3), '3 is lesser than or equal to 3');
  t.ok(is.le('a', 'abc'), 'a is lesser than or equal to abc');
  t.ok(is.le('abc', 'abc'), 'abc is lesser than or equal to abc');
  t.notOk(is.le('abc', 'a'), 'abc is not lesser than or equal to a');
  t.notOk(is.le(Infinity, 0), 'infinity is not lesser than or equal to anything');
  t.notOk(is.le(0, Infinity), 'anything is not lesser than or equal to infinity');
  var error = new TypeError('NaN is not a valid value');
  t.throws(function () { return is.le(NaN, 2); }, error, 'throws when first value is NaN');
  t.throws(function () { return is.le(2, NaN); }, error, 'throws when second value is NaN');
  t.end();
});

test('is.lt', function (t) {
  t.ok(is.lt(2, 3), '2 is lesser than 3');
  t.notOk(is.lt(3, 2), '3 is not lesser than 2');
  t.notOk(is.lt(3, 3), '3 is not lesser than 3');
  t.ok(is.lt('a', 'abc'), 'a is lesser than abc');
  t.notOk(is.lt('abc', 'abc'), 'abc is not lesser than abc');
  t.notOk(is.lt('abc', 'a'), 'abc is not lesser than a');
  t.notOk(is.lt(Infinity, 0), 'infinity is not lesser than anything');
  t.notOk(is.lt(0, Infinity), 'anything is not lesser than infinity');
  var error = new TypeError('NaN is not a valid value');
  t.throws(function () { return is.lt(NaN, 2); }, error, 'throws when first value is NaN');
  t.throws(function () { return is.lt(2, NaN); }, error, 'throws when second value is NaN');
  t.end();
});

test('is.within', function (t) {
  var nanError = new TypeError('NaN is not a valid value');
  t.throws(function () { return is.within(NaN, 0, 0); }, nanError, 'throws when first value is NaN');
  t.throws(function () { return is.within(0, NaN, 0); }, nanError, 'throws when second value is NaN');
  t.throws(function () { return is.within(0, 0, NaN); }, nanError, 'throws when third value is NaN');

  var error = new TypeError('all arguments must be numbers');
  t.throws(function () { return is.within('', 0, 0); }, error, 'throws when first value is string');
  t.throws(function () { return is.within(0, '', 0); }, error, 'throws when second value is string');
  t.throws(function () { return is.within(0, 0, ''); }, error, 'throws when third value is string');
  t.throws(function () { return is.within({}, 0, 0); }, error, 'throws when first value is object');
  t.throws(function () { return is.within(0, {}, 0); }, error, 'throws when second value is object');
  t.throws(function () { return is.within(0, 0, {}); }, error, 'throws when third value is object');
  t.throws(function () { return is.within(null, 0, 0); }, error, 'throws when first value is null');
  t.throws(function () { return is.within(0, null, 0); }, error, 'throws when second value is null');
  t.throws(function () { return is.within(0, 0, null); }, error, 'throws when third value is null');
  t.throws(function () { return is.within(undefined, 0, 0); }, error, 'throws when first value is undefined');
  t.throws(function () { return is.within(0, undefined, 0); }, error, 'throws when second value is undefined');
  t.throws(function () { return is.within(0, 0, undefined); }, error, 'throws when third value is undefined');

  t.ok(is.within(2, 1, 3), '2 is between 1 and 3');
  t.ok(is.within(0, -1, 1), '0 is between -1 and 1');
  t.ok(is.within(2, 0, Infinity), 'infinity always returns true');
  t.ok(is.within(2, Infinity, 2), 'infinity always returns true');
  t.ok(is.within(Infinity, 0, 1), 'infinity always returns true');
  t.notOk(is.within(2, -1, -1), '2 is not between -1 and 1');
  t.end();
});

test('is.object', function (t) {
  t.ok(is.object({}), 'object literal is object');
  t.notOk(is.object(), 'undefined is not an object');
  t.notOk(is.object(null), 'null is not an object');
  t.notOk(is.object(true), 'true is not an object');
  t.notOk(is.object(''), 'string is not an object');
  t.notOk(is.object(NaN), 'NaN is not an object');
  t.notOk(is.object(Object), 'object constructor is not an object');
  t.notOk(is.object(function () {}), 'function is not an object');
  t.end();
});

test('is.hash', function (t) {
  t.ok(is.hash({}), 'empty object literal is hash');
  t.ok(is.hash({ 1: 2, a: "b" }), 'object literal is hash');
  t.notOk(is.hash(), 'undefined is not a hash');
  t.notOk(is.hash(null), 'null is not a hash');
  t.notOk(is.hash(new Date()), 'date is not a hash');
  t.notOk(is.hash(new String()), 'string object is not a hash');
  t.notOk(is.hash(''), 'string literal is not a hash');
  t.notOk(is.hash(new Number()), 'number object is not a hash');
  t.notOk(is.hash(1), 'number literal is not a hash');
  t.notOk(is.hash(true), 'true is not a hash');
  t.notOk(is.hash(false), 'false is not a hash');
  t.notOk(is.hash(new Boolean()), 'boolean obj is not hash');
  t.notOk(is.hash(false), 'literal false is not hash');
  t.notOk(is.hash(true), 'literal true is not hash');
  if (typeof module !== 'undefined') {
    t.ok(is.hash(module.exports), 'module.exports is a hash');
  }
  if (typeof window !== 'undefined') {
    t.notOk(is.hash(window), 'window is not a hash');
    t.notOk(is.hash(document.createElement('div')), 'element is not a hash');
  } else if (typeof process !== 'undefined') {
    t.notOk(is.hash(global), 'global is not a hash');
    t.notOk(is.hash(process), 'process is not a hash');
  }
  t.end();
});

test('is.regexp', function (t) {
  t.ok(is.regexp(/a/g), 'regex literal is regex');
  t.ok(is.regexp(new RegExp('a', 'g')), 'regex object is regex');
  t.notOk(is.regexp(), 'undefined is not regex');
  t.notOk(is.regexp(function () {}), 'function is not regex');
  t.notOk(is.regexp('/a/g'), 'string regex is not regex');
  t.end();
});

test('is.string', function (t) {
  t.ok(is.string('foo'), 'string literal is string');
  t.ok(is.string(new String('foo')), 'string literal is string');
  t.notOk(is.string(), 'undefined is not string');
  t.notOk(is.string(String), 'string constructor is not string');
  var F = function () {};
  F.prototype = new String();
  t.notOk(is.string(F), 'string subtype is not string');
  t.end();
});

