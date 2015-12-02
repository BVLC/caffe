var test = require('tap').test;
var gyp = require('..');

test("Conditional sections", function(t) {
    var tests = 2;
    gyp({ a: 1, conditions: [["merge==1", { b: 2}]] }, {merge: 1}, function(err, out) {
        t.equal(out.a, "1", "a should be 1");
        t.equal(out.b, "2", "b should be 2");
        if (--tests == 0) t.end();
    });
    gyp({ a: 1, conditions: [["merge==2", { b: 2}]] }, {merge: 1}, function(err, out) {
        t.equal(out.a, "1", "a should be 1");
        t.not(out.b, "2", "b should not be 2");
        if (--tests == 0) t.end();
    });
});
