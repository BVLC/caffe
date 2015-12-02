require('ometajs');
var cond = require('../index.ometajs');
var test = require('tap').test;

test("Arithmetic", function (t) {
    t.equals(cond('1+1'), 2, "Addition");
    t.equals(cond('1-1'), 0, "Subtraction");
    t.equals(cond('1 + -1'), 0, "Addition of negative number");
    t.equals(cond('1 * 2'), 2, "Multiplication");
    t.equals(cond('2 + 2 * 3'), 8, "Multiplication with precedence");
    t.end();
});

test("Bitwise math", function (t) {
    t.equals(cond('1 & 3'), 1, "And: 1 & 3");
    t.equals(cond('2 & 3'), 2, "And: 2 & 3");
    t.equals(cond('2 & 1'), 0, "And: 2 & 1");
    t.equals(cond('1 | 1'), 1, "Or: 1 | 1");
    t.equals(cond('1 | 2'), 3, "Or: 1 | 2");
    t.equals(cond('1 ^ 1'), 0, "Xor: 1 ^ 1");
    t.equals(cond('1 ^ 2'), 3, "Xor: 1 ^ 2");
    t.equals(cond('1 << 1'), 2, "Shift: 1 << 1");
    t.equals(cond('2 >> 1'), 1, "Shift: 2 >> 1");
    t.end();
});
