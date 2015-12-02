var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['product'] = function () {
    var lazy = new Lazy;
    var executed = 0;
    lazy.product(function (y) {
        executed++;
        assert.equal(y, 1*2*3*4*5*6*7*8*9);
    })

    range(1,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('end');

    assert.equal(executed, 1, 'product failed to execute');
}

