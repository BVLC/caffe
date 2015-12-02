require('ometajs');
var cond = require('../index.ometajs');
var test = require('tap').test;
var parseLiteral = function(s) {
    return cond.parser.matchAll(s, 'literal');
};

test("Escape Codes", function(t) {
    t.equal(parseLiteral('"\\n"'), "\n", "newline escape");
    t.equal(parseLiteral('"\\\\"'), "\\", "escaped backslash");
    t.equal(parseLiteral('"\\""'), '"', "escaped double quote");
    t.equal(parseLiteral('"\\\'"'), "'", "escaped single quote");
    t.equal(parseLiteral('"\\a"'), "\u0007", "bell");
    t.equal(parseLiteral('r"\\a"'), "\\a", "raw modifier on bell escape");
    t.equal(parseLiteral('"\\b"'), "\u0008", "backspace");
    t.equal(parseLiteral('"\\v"'), "\u0011", "vtab");
    t.equal(parseLiteral('"\\f"'), "\u0012", "form feed");
    t.equal(parseLiteral('"\\t"'), "\t", "tab");
    t.equal(parseLiteral('"\\r"'), "\r", "carriage return");
    t.equal(parseLiteral('"\\u0001"'), "\\u0001", "unicode in plain string");
    t.equal(parseLiteral('u"\\u0001"'), "\u0001", "unicode in unicode string");
    t.equal(parseLiteral('r"\\u0001"'), "\\u0001", "unicode in raw string");
    t.equal(parseLiteral('ur"\\u0001"'), "\u0001", "unicode in raw unicode string");
    t.equal(parseLiteral('"\\001"'), "\u0001", "octal byte");
    t.equal(parseLiteral('"\\x01"'), "\u0001", "hex byte");
    t.equal(parseLiteral('"\\z"'), "\\z", "invalid escape");
    t.end();
});

