var assert = require('assert');
var Lazy = require('..');
var EventEmitter = require('events').EventEmitter;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports.em = function () {
    var em = new EventEmitter;
    var i = 0;
    var iv = setInterval(function () {
        em.emit('data', i++);
    }, 10);
    
    var caughtDone = 0;
    em.on('pipe', function () { caughtDone ++ });
    
    var joined = 0;
    Lazy(em).take(10).join(function (xs) {
        assert.deepEqual(xs, range(0, 10));
        clearInterval(iv);
        joined ++;
    });
    
    setTimeout(function () {
        assert.equal(joined, 1);
        assert.equal(caughtDone, 1);
    }, 500);
}

