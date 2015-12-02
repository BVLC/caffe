var test = require('tap').test;
var gyp = require('..');

test("simple expansions", function(t) {
    gyp({ a: 1, b: "<(c)" }, {c: 2}, function(err, out) {
        t.equal(out.a, "1", "a should be 1");
        t.equal(out.b, "2", "b should be 2");
        t.end();
    });
});

test("variable section expansions", function(t) {
    gyp({ a: 1, b: "<(c)", variables: { c: 2 } }, {}, function(err, out) {
        t.equal(out.a, "1", "a should be 1");
        t.equal(out.b, "2", "b should be 2");
        t.end();
    });
});

test("list expansion", function(t) {
    gyp({ a: [ "<@(v)" ], b: [ "<@(l)" ] }, {l: [1, 2], v: "1"}, function(err, out) {
        t.equal(out.a[0], "1", "a[0] should be 1");
        t.equal(out.b[0], "1", "b[0] should be 1");
        t.equal(out.b[1], "2", "b[1] should be 2");
        t.end();
    });
});

