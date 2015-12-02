var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['head'] = function () {
    var lazy = new Lazy;
    var data = [];
    var executed = 0;
    lazy.head(function (x) {
        assert.equal(x, 0);
        executed++;
    })

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });

    assert.equal(executed, 1, 'head executed too much');
}

