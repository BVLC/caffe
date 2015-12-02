var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['join'] = function () {
    var lazy = new Lazy;
    var data = [];
    var executed = false;
    lazy.take(10).join(function (xs) {
        assert.deepEqual(xs, range(0,10));
        executed = true;
    });

    range(0,20).forEach(function (x) {
        lazy.emit('data', x);
    });

    assert.ok(executed, 'join didn\'t execute');
}

