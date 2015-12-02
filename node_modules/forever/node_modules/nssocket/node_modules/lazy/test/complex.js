var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['complex'] = function () {
    var lazy = new Lazy;
    var data1 = [];
    var data2 = [];
    var data3 = [];
    var joinExecuted = false;
    lazy
        .map(function (x) {
            return 2*x;
        })
        .forEach(function (x) {
            data1.push(x);
        })
        .map(function (x) {
            return x/2;
        })
        .take(5)
        .forEach(function (x) {
            data2.push(x);
        })
        .filter(function (x) {
            return x % 2 == 0;
        })
        .forEach(function (x) {
            data3.push(x);
        })
        .join(function (xs) {
            joinExecuted = true;
            assert.deepEqual(xs, [0, 2, 4]);
        });

    range(0,10).forEach(function (x) {
        lazy.emit('data', x);
    });
    lazy.emit('end');

    assert.deepEqual(data1, range(0,10).map(function (x) { return x*2 }));
    assert.deepEqual(data2, range(0,5));
    assert.deepEqual(data3, [0, 2, 4]);
    assert.ok(joinExecuted);
}

