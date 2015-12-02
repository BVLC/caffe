var tap = require('tap');
var spawn = require('child_process').spawn;

tap.test('exit ok', function (t) {
    t.plan(2);
    
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
        t.same(rs, [
            'TAP version 13',
            'array',
            { id: 1, ok: true, name: 'should be equivalent' },
            { id: 2, ok: true, name: 'should be equivalent' },
            { id: 3, ok: true, name: 'should be equivalent' },
            { id: 4, ok: true, name: 'should be equivalent' },
            { id: 5, ok: true, name: 'should be equivalent' },
            'tests 5',
            'pass  5',
            'ok'
        ]);
    });
    
    var ps = spawn(process.execPath, [ __dirname + '/exit/ok.js' ]);
    ps.stdout.pipe(tc);
    ps.on('exit', function (code) {
        t.equal(code, 0);
    });
});

tap.test('exit fail', function (t) {
    t.plan(2);
    
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
        t.same(rs, [
            'TAP version 13',
            'array',
            { id: 1, ok: true, name: 'should be equivalent' },
            { id: 2, ok: true, name: 'should be equivalent' },
            { id: 3, ok: true, name: 'should be equivalent' },
            { id: 4, ok: true, name: 'should be equivalent' },
            { id: 5, ok: false, name: 'should be equivalent' },
            'tests 5',
            'pass  4',
            'fail  1'
        ]);
    });
    
    var ps = spawn(process.execPath, [ __dirname + '/exit/fail.js' ]);
    ps.stdout.pipe(tc);
    ps.on('exit', function (code) {
        t.notEqual(code, 0);
    });
});

tap.test('too few exit', function (t) {
    t.plan(2);
    
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
        t.same(rs, [
            'TAP version 13',
            'array',
            { id: 1, ok: true, name: 'should be equivalent' },
            { id: 2, ok: true, name: 'should be equivalent' },
            { id: 3, ok: true, name: 'should be equivalent' },
            { id: 4, ok: true, name: 'should be equivalent' },
            { id: 5, ok: true, name: 'should be equivalent' },
            { id: 6, ok: false, name: 'plan != count' },
            'tests 6',
            'pass  5',
            'fail  1'
        ]);
    });
    
    var ps = spawn(process.execPath, [ __dirname + '/exit/too_few.js' ]);
    ps.stdout.pipe(tc);
    ps.on('exit', function (code) {
        t.notEqual(code, 0);
    });
});

tap.test('more planned in a second test', function (t) {
    t.plan(2);
    
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
        t.same(rs, [
            'TAP version 13',
            'first',
            { id: 1, ok: true, name: '(unnamed assert)' },
            'second',
            { id: 2, ok: true, name: '(unnamed assert)' },
            { id: 3, ok: false, name: 'plan != count' },
            'tests 3',
            'pass  2',
            'fail  1'
        ]);
    });
    
    var ps = spawn(process.execPath, [ __dirname + '/exit/second.js' ]);
    ps.stdout.pipe(tc);
    ps.on('exit', function (code) {
        t.notEqual(code, 0);
    });
});
