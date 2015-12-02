require('ometajs');
var cond = require('../index.ometajs');
var test = require('tap').test;

test("Basic conditions", function (t) {
    t.equal(cond('n>0', {n: 1}), true);
    t.equal(cond('n<1', {n: 1}), false);
    t.equal(cond('n<=1', {n: 1}), true);
    t.end();
});

test("If expression", function (t) {
    t.equals(cond("1 if 1 else 2"), 1, "If true");
    t.equals(cond("1 if 0 else 2"), 2, "If false");
    t.end();
});
