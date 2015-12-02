var test = require('tap').test;
var gyp = require('..');

test("parse simple dictionary", function(t) {
    gyp({ a: 1, b: "test" }, {}, function(err, out) {
        t.equal(out.a, "1", "a should be 1");
        t.equal(out.b, "test", "b should 'test'");
        t.end();
    });
});

test("parse from file", function(t) {
    gyp(__dirname + "/test.gyp", {}, function(err, out) {
        t.equal(out.a, "1", "a should be 1");
        t.equal(out.b, "test", "b should 'test'");
        t.end();
    });
});
