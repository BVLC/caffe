var test = require('tape');
var through = require('through2');
var EventEmitter = require('events').EventEmitter;
var wrap = require('../');

test('falsey object mode', function (t) {
    var oldStream = new EventEmitter;
    var wrapped = wrap.obj(oldStream);
    
    var input = [ 5, 'a', false, 0, '', 'xyz', { x: 4 }, 7, [], 555 ];
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

test('falsey object mode option', function (t) {
    var oldStream = new EventEmitter;
    var wrapped = wrap(oldStream, { objectMode: true });
    
    var input = [ 5, 'a', false, 0, '', 'xyz', { x: 4 }, 7, [], 555 ];
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
