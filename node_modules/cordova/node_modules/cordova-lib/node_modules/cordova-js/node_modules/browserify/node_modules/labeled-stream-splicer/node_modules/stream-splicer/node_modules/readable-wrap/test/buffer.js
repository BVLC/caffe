var test = require('tape');
var through = require('through2');
var EventEmitter = require('events').EventEmitter;
var wrap = require('../');

test('buffer', function (t) {
    var oldStream = new EventEmitter;
    var wrapped = wrap(oldStream);
    
    var input = [ Buffer('abc'), Buffer('def'), Buffer('ghi') ];
    var expected = input.slice();
    t.plan(expected.length + 1);
    
    wrapped.pipe(through.obj(write, end));
    
    var iv = setInterval(function () {
        if (input.length === 0) {
            oldStream.emit('end');
            clearInterval(iv);
        }
        else oldStream.emit('data', input.shift());
    }, 5);
    
    function write (row, enc, next) {
        t.deepEqual(row, expected.shift());
        next();
    }
    
    function end () { t.ok('ended') }
});
