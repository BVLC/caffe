var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['foldr'] = function () {
    var lazy = new Lazy;
    var executed = 0;
    lazy.foldr(function (x, acc) { return acc + x; }, 0, function (y) {
        executed++;
        assert.equal(y, 45);
    })

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('end');

    assert.equal(executed, 1, 'foldr failed to execute');
}

