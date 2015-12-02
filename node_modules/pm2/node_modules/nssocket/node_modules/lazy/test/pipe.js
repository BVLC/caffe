var assert = require('assert');
var Lazy = require('..');
var EventEmitter = require('events').EventEmitter;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports.pipe = function () {
    var em = new EventEmitter;
    var i = 0;
    var iv = setInterval(function () {
        em.emit('data', i++);
    }, 10);
    
    var caught = { pipe : 0, end : 0 };
    em.on('pipe', function () {
        caught.pipe ++;
        setTimeout(em.emit.bind(em, 'end'), 50);
    });
    em.on('end', function () { caught.end ++ });
    
    var joined = 0;
    Lazy(em).take(10).join(function (xs) {
        assert.deepEqual(xs, range(0, 10));
        clearInterval(iv);
        joined ++;
    });
    
    setTimeout(function () {
        assert.equal(joined, 1);
        assert.equal(caught.pipe, 1);
        assert.equal(caught.end, 1);
    }, 1000);
}

