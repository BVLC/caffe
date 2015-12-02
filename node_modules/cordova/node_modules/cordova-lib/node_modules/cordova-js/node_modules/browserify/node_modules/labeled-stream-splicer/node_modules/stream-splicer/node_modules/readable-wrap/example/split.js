var split = require('split');
var wrap = require('../');
var through = require('through2');

process.stdin.pipe(wrap.obj(split())).pipe(through.obj(write));

function write (buf, enc, next) {
    console.log(buf.length + ': ' + buf);
    next();
}
