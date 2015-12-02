require('ometajs');
var cond = require('../index.ometajs');
var test = require('tap').test;
var parseLiteral = function(s) {
    return cond.parser.matchAll(s, 'literal');
};

test("Literals", function(t) {
    t.equal(parseLiteral('0'), 0, "Zero");
    t.equal(parseLiteral('010'), 8, "Octal 10 with 0 prefix");
    t.equal(parseLiteral('0o10'), 8, "Octal 10 with 0o prefix");
    t.equal(parseLiteral('0x10'), 16, "Hex 10");
    t.equal(parseLiteral('2.0'), 2.0, "Float 2.0");
    t.equal(parseLiteral('2.5'), 2.5, "Float 2.5");
    t.equal(parseLiteral('2.5e5'), 250000, "2.5e5");
    t.equal(parseLiteral('2.5e-1'), 0.25, "2.5e-1");
    t.equal(parseLiteral('"test"'), "test", "double quoted string");
    t.equal(parseLiteral("'test'"), "test", "single quoted string");
    t.equal(parseLiteral('"""test"""'), "test", "double quoted longstring");
    t.equal(parseLiteral("'''test'''"), "test", "single quoted longstring");
    t.end();
});

