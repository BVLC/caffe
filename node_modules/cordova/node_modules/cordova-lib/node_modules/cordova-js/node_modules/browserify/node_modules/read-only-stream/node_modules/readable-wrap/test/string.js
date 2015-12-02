var test = require('tape');
var through = require('through2');
var EventEmitter = require('events').EventEmitter;
var wrap = require('../');

test('string', function (t) {
    var oldStream = new EventEmitter;
    var wrapped = wrap(oldStream);
    
    var input = [ 'abc', 'def', 'ghi' ];
    var expected = [];
    for (var i = 0; i < input.length; i++) {
        expected.push(Buffer(input[i]));
    }
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
