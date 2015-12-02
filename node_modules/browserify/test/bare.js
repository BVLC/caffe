var test = require('tap').test;
var spawn = require('child_process').spawn;
var path = require('path');
var concat = require('concat-stream');
var vm = require('vm');

test('bare', function (t) {
    t.plan(4);
    
    var cwd = process.cwd();
    process.chdir(__dirname);
    
    var ps = spawn(process.execPath, [
        path.resolve(__dirname, '../bin/cmd.js'),
        '-', '--bare'
    ]);
    ps.stdout.pipe(concat(function (body) {
        vm.runInNewContext(body, {
            Buffer: function (s) { return s.toLowerCase() },
            console: {
                log: function (msg) { t.equal(msg, 'abc') }
            }
        });
        vm.runInNewContext(body, {
            Buffer: Buffer,
            console: {
                log: function (msg) {
                    t.ok(Buffer.isBuffer(msg));
                    t.equal(msg.toString('utf8'), 'ABC')
                }
            }
        });
    }));
    ps.stdin.end('console.log(Buffer("ABC"))');
    
    ps.on('exit', function (code) {
        t.equal(code, 0);
    });
});

test('bare inserts __filename,__dirname but not process,global,Buffer', function (t) {
    t.plan(2);
    
    var ps = spawn(process.execPath, [
        path.resolve(__dirname, '../bin/cmd.js'),
        path.resolve(__dirname, 'bare/main.js'),
        '--bare'
    ]);
    
    ps.stdout.pipe(concat(function (body) {
        vm.runInNewContext(body, {
            console: {
                log: function (msg) {
                    t.same(msg, [
                        path.join(__dirname, 'bare'),
                        path.join(__dirname, 'bare/main.js'),
                        'undefined',
                        'undefined',
                        'undefined'
                    ]);
                }
            }
        });
    }));
    ps.stdin.end();
    
    ps.on('exit', function (code) {
        t.equal(code, 0);
    });
});
