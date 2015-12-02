var test = require('../');

function fn() {
    throw new TypeError('RegExp');
}

test('throws', function (t) {
    t.throws(fn);
    t.end();
});

test('throws (RegExp match)', function (t) {
    t.throws(fn, /RegExp/);
    t.end();
});

test('throws (Function match)', function (t) {
    t.throws(fn, TypeError);
    t.end();
});
