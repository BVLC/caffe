var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['map'] = function () {
    var lazy = new Lazy;
    var executed = false;
    var data = [];
    lazy
        .map(function (x) {
            return 2*x;
        })
        .forEach(function (x) {
            data.push(x);
        });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });

    assert.deepEqual(data, range(0,10).map(function (x) { return x*2 }));
}

