var assert = require('assert');
var Lazy = require('..');

exports.bucket = function () {
    var joined = false;
    var lazy = new Lazy;
    lazy.bucket('', function splitter (acc, x) {
        var accx = acc + x;
        var i = accx.indexOf('\n');
        if (i >= 0) {
            this(accx.slice(0, i));
            return splitter.call(this, accx.slice(i + 1), '');
        }
        return accx;
    }).join(function (lines) {
        assert.deepEqual(lines, 'foo bar baz quux moo'.split(' '));
        joined = true;
    });
    
    setTimeout(function () {
        lazy.emit('data', 'foo\nba');
    }, 50);
    setTimeout(function () {
        lazy.emit('data', 'r');
    }, 100);
    setTimeout(function () {
        lazy.emit('data', '\nbaz\nquux\nm');
    }, 150);
    setTimeout(function () {
        lazy.emit('data', 'oo');
    }, 200);
    setTimeout(function () {
        lazy.emit('data', '\nzoom');
        lazy.emit('end');
        assert.ok(joined);
    }, 250);
};
