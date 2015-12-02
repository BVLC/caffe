var tape = require('../');
var tap = require('tap');

tap.test('nested sync test without plan or end', function (tt) {
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
        var expected = [
            'TAP version 13',
            'nested without plan or end',
            'first',
            { id: 1, ok: true, name: '(unnamed assert)' },
            'second',
            { id: 2, ok: true, name: '(unnamed assert)' },
            'tests 2',
            'pass  2',
            'ok'
        ]
        tt.same(rs, expected);
    });

    test.createStream().pipe(tc);

    test('nested without plan or end', function(t) {
        t.test('first', function(q) {
            setTimeout(function first() { 
                q.ok(true);
                q.end() 
            }, 10);
        });
        t.test('second', function(q) {
            setTimeout(function second() { 
                q.ok(true);
                q.end() 
            }, 10);
        });
    });

});
