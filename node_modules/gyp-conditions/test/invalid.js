require('ometajs');
var cond = require('../index.ometajs');
var test = require('tap').test;
var parseExpr = function(s) {
    return cond.parser.matchAll(s, 'expr');
};

test("Fail on invalid", function(t) {
    t.throws(function() {
        parseExpr("1hello");
    });
    t.throws(function() {
        parseExpr("hello1");
    });
    t.throws(function() {
        parseExpr("hello 1");
    });
    t.throws(function() {
        parseExpr("1 hello");
    });
    t.end();
});

