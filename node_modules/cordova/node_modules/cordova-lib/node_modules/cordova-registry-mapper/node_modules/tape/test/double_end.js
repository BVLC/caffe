var test = require('tap').test;
var concat = require('concat-stream');
var spawn = require('child_process').spawn;

test(function (t) {
    t.plan(2);
    var ps = spawn(process.execPath, [ __dirname + '/double_end/double.js' ]);
    ps.on('exit', function (code) {
        t.equal(code, 1);
    });
    ps.stdout.pipe(concat(function (body) {
        t.equal(body.toString('utf8'), [
            'TAP version 13',
            '# double end',
            'ok 1 should be equal',
            'not ok 2 .end() called twice',
            '  ---',
            '    operator: fail',
            '  ...',
            '',
            '1..2',
            '# tests 2',
            '# pass  1',
            '# fail  1',
        ].join('\n') + '\n\n');
    }));
});
