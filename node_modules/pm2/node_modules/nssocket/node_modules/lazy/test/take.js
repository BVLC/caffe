var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['take'] = function () {
    var lazy = new Lazy;
    var data = [];
    var executed = 0;
    lazy.take(6).join(function (xs) {
        assert.deepEqual(xs, range(0,6));
        executed++;
    });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });

    assert.equal(executed, 1, 'join executed incorrectly');
}

