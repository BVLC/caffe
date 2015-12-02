var test = require('tap').test;
var gyp = require('..');

test("parse simple dictionary", function(t) {
    var out = gyp('simple.gyp', __dirname);
    t.equal(out.a, 1, "a should be 1");
    t.equal(out.b, "test", "b should 'test'");
    t.end();
});

test("simple include", function (t) {
    var out = gyp('root.gyp', __dirname);
    t.equal(out.included, 1, 'included should be 1');
    t.end();
});

test("include in dictionary", function (t) {
    var out = gyp('in-dict.gyp', __dirname);
    t.equal(out.a.included, 1, 'included should be 1');
    t.end();
});

test("include in array", function (t) {
    var out = gyp('in-array.gyp', __dirname);
    t.equal(out.a[0].included, 1, 'included should be 1');
    t.end();
});

test("merge while including", function (t) {
    var out = gyp('merge.gyp', __dirname);
    t.equal(out.included, 1, 'included should be 1');
    t.end();
});
