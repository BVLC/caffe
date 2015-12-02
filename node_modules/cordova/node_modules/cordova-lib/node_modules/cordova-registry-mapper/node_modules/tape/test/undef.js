var tape = require('../');
var tap = require('tap');
var concat = require('concat-stream');

tap.test('array test', function (tt) {
    tt.plan(1);
    
    var test = tape.createHarness();
    test.createStream().pipe(concat(function (body) {
        tt.equal(
            body.toString('utf8'),
            'TAP version 13\n'
            + '# undef\n'
            + 'not ok 1 should be equivalent\n'
            + '  ---\n'
            + '    operator: deepEqual\n'
            + '    expected: { beep: undefined }\n'
            + '    actual:   {}\n'
            + '  ...\n'
            + '\n'
            + '1..1\n'
            + '# tests 1\n'
            + '# pass  0\n'
            + '# fail  1\n'
        );
    }));
    
    test('undef', function (t) {
        t.plan(1);
        t.deepEqual({}, { beep: undefined });
    });
});
