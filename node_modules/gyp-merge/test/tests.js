var test = require('tap').test;
var merge = require('..');

test("simple dictionary vs simple dictionary", function(t) {
    var out = merge( { a: 1, b: "test" }, { a: 2, c: "new" });
    t.equal(out.a, 2, "a should be replaced with 2");
    t.equal(out.b, "test", "b should be replaced with 'test'");
    t.equal(out.c, "new", "c should be 'new'");
    t.end();
});

test("recursive dictionary merge", function(t) {
    var out = merge( { a: { b: { c: 1 } } }, { a: { b: { d: 2 }, f: 1 }, e: 3 } );
    t.equal(out.a.b.c, 1, "a.b.c should remain 1");
    t.equal(out.a.b.d, 2, "a.b.d should have been added as 2");
    t.equal(out.e, 3, "e should have been added as 3");
    t.equal(out.a.f, 1, "a.f should have been added as 1");
    t.end();
});

test("dictionary with list merges", function(t) {
    var out = merge(
        { a: [1, 2, 3], b: [4, 5, 6], c: [7, 8, 9], d: [10, 11, 12] },
        { a: ["A", "B", "C"], "b?": ["D", "E", "F"], "c+": ["G", "H", "I"], "d=" : ["J", "K", "L" ]}
    );

    t.deepEqual( out.a, [1, 2, 3, "A", "B", "C"], "a should be appended");
    t.deepEqual( out.b, [4, 5, 6], "b should be left alone");
    t.deepEqual( out.c, ["G", "H", "I", 7, 8, 9], "c should be prepended");
    t.deepEqual( out.d, ["J", "K", "L"], "d should be replaced");

    t.end();
});

test("handle singletons in list merges", function (t) {
    var out = merge(
        {
            'defines': [
                'NDEBUG',
                'USE_THREADS'
            ]
        },
        {
            'defines': [
                'EXPERIMENT=1',
                'NDEBUG'
            ]
        }
    );

    t.deepEqual(out, {
        'defines': [
            'NDEBUG',
            'USE_THREADS',
            'EXPERIMENT=1'
        ]
    });

    t.end();
});

test("avoid handling singletons in list merges", function (t) {
    var out = merge(
        {
            'defines': [
                'NDEBUG',
                'USE_THREADS'
            ]
        },
        {
            'defines': [
                'EXPERIMENT=1',
                'NDEBUG'
            ]
        },
        { noSingletons: true }
    );

    t.deepEqual(out, {
        'defines': [
            'NDEBUG',
            'USE_THREADS',
            'EXPERIMENT=1',
            'NDEBUG'
        ]
    });

    t.end();
});
