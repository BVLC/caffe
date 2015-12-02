var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['sum'] = function () {
    var lazy = new Lazy;
    var executed = 0;
    lazy.sum(function (y) {
        executed++;
        assert.equal(y, 45);
    })

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('end');

    assert.equal(executed, 1, 'sum failed to execute');
}

