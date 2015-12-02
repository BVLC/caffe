var falafel = require('falafel');
var tape = require('../');
var tap = require('tap');

tap.test('array test', function (tt) {
    tt.plan(1);
    
    var test = tape.createHarness({ exit : false });
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
    
    test.createStream().pipe(tc);
    
    test('array', function (t) {
        t.plan(5);
        
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
                t.same(xs, [ [ 1, 2, [ 3, 4444 ] ], [ 5, 6 ] ]);
            }
        );
    });
});
