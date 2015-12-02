var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['forEach'] = function () {
    var lazy = new Lazy;
    var executed = false;
    var data1 = [];
    var data2 = [];
    lazy
        .forEach(function (x) {
            data1.push(x);
        })
        .forEach(function (x) {
            data2.push(x);
        });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });

    assert.deepEqual(data1, range(0,10));
    assert.deepEqual(data2, range(0,10));
}

