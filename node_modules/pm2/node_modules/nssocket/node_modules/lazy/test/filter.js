var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['filter'] = function () {
    var lazy = new Lazy;
    var data = [];
    var executed = false;
    lazy.filter(function(x) { return x > 5 }).join(function (xs) {
        assert.deepEqual(xs, [6,7,8,9]);
        executed = true;
    });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('pipe');

    assert.ok(executed, 'join didn\'t execute');
}

