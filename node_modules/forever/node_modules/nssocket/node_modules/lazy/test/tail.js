var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['tail'] = function () {
    var lazy = new Lazy;
    var data = [];
    var executed = false;
    lazy.tail().join(function (xs) {
        assert.deepEqual(xs, range(1,10));
        executed = true;
    });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('end');

    assert.ok(executed, 'join didn\'t execute');
}

