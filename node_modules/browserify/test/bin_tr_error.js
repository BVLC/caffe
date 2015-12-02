var browserify = require('../');
var spawn = require('child_process').spawn;
var test = require('tap').test;
var path = require('path')

test('function transform', function (t) {
    t.plan(3);
    var ps = spawn(process.execPath, [
        path.resolve(__dirname, '../bin/cmd.js'),
        '-t', './tr.js', './main.js'
    ], {cwd: path.resolve(__dirname, 'bin_tr_error')});
    var src = '';
    var err = '';
    ps.stdout.on('data', function (buf) { src += buf });
    ps.stderr.on('data', function (buf) { err += buf });
    
    ps.on('exit', function (code) {
        t.notEqual(code, 0);
        var errorFile = path.resolve(__dirname, 'bin_tr_error', 'tr.js');
        t.notEqual(err.indexOf('there was error'), -1, 'Error should contain error message')
        t.notEqual(err.indexOf(errorFile), -1, 'Error should contain stack trace')
    });
});
