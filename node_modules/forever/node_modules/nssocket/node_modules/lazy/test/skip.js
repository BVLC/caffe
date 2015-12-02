var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['skip'] = function () {
    var lazy = new Lazy;
    var data = [];
    var executed = 0;
    lazy.skip(6).join(function (xs) {
        assert.deepEqual(xs, range(6,10));
        executed++;
    });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('end');

    assert.equal(executed, 1, 'join executed incorrectly');
}

