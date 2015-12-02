var falafel = require('falafel');
var tape = require('../');
var tap = require('tap');

tap.test('array test', function (tt) {
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
            'nested array test',
            { id: 1, ok: true, name: 'should be equivalent' },
            { id: 2, ok: true, name: 'should be equivalent' },
            { id: 3, ok: true, name: 'should be equivalent' },
            { id: 4, ok: true, name: 'should be equivalent' },
            { id: 5, ok: true, name: 'should be equivalent' },
            'inside test',
            { id: 6, ok: true, name: '(unnamed assert)' },
            { id: 7, ok: true, name: '(unnamed assert)' },
            'another',
            { id: 8, ok: true, name: '(unnamed assert)' },
            'tests 8',
            'pass  8',
            'ok'
        ]);
    });
    
    test.createStream().pipe(tc);
    
    test('nested array test', function (t) {
        t.plan(6);
        
        var src = '(' + function () {
            var xs = [ 1, 2, [ 3, 4 ] ];
            var ys = [ 5, 6 ];
            g([ xs, ys ]);
        } + ')()';
        
        var output = falafel(src, function (node) {
            if (node.type === 'ArrayExpression') {
                node.update('fn(' + node.source() + ')');
            }
        });
        
        t.test('inside test', function (q) {
            q.plan(2);
            q.ok(true);
            
            setTimeout(function () {
                q.ok(true);
            }, 100);
        });
        
        var arrays = [
            [ 3, 4 ],
            [ 1, 2, [ 3, 4 ] ],
            [ 5, 6 ],
            [ [ 1, 2, [ 3, 4 ] ], [ 5, 6 ] ],
        ];
        
        Function(['fn','g'], output)(
            function (xs) {
                t.same(arrays.shift(), xs);
                return xs;
            },
            function (xs) {
                t.same(xs, [ [ 1, 2, [ 3, 4 ] ], [ 5, 6 ] ]);
            }
        );
    });

    test('another', function (t) {
        t.plan(1);
        setTimeout(function () {
            t.ok(true);
        }, 50);
    });
});
