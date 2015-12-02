var assert = require('assert');
var Lazy = require('..');
var expresso = expresso;

function range(i, j) {
    var r = [];
    for (;i<j;i++) r.push(i);
    return r;
}

exports['custom event names'] = function () {
    var lazy = new Lazy(null, { data : 'meow', pipe : 'all done' });
    var data = [];
    var executed = false;
    lazy.take(10).join(function (xs) {
        assert.deepEqual(xs, range(0,10));
        executed = true;
    });
    
    var allDone = 0;
    lazy.on('all done', function () {
        allDone ++;
    });

    range(0,20).forEach(function (x) {
        lazy.emit('meow', x);
    });

    assert.ok(executed, 'join didn\'t execute');
    assert.equal(allDone, 1);
}

