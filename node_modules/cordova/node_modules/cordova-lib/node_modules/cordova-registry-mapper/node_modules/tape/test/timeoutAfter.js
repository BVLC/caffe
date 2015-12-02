var tape = require('../');
var tap = require('tap');

tap.test('timeoutAfter test', function (tt) {
    tt.plan(1);
    
    var test = tape.createHarness();
    var tc = tap.createConsumer();
    
    var rows = [];
    tc.on('data', function (r) { rows.push(r) });
    tc.on('end', function () {
        var rs = rows.map(function (r) {
            if (r && typeof r === 'object') {
                return { id : r.id, ok : r.ok, name : r.name.trim() };
            }
            else return r;
        });
        tt.same(rs, [
            'TAP version 13',
            'timeoutAfter',
            { id: 1, ok: false, name: 'test timed out after 1ms' },
            'tests 1',
            'pass  0',
            'fail  1'
        ]);
    });
    
    test.createStream().pipe(tc);
    
    test('timeoutAfter', function (t) {
        t.plan(1);
        t.timeoutAfter(1);
    });
});
