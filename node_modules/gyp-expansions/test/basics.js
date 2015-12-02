var test = require('tap').test;
var expansions = require('..');

test('A string with no expansions returns the string unmodified', function(t) {
    t.plan(1);
    expansions.expandString('a', {}, 'pre', function(e, r) {
        t.equal(r, 'a');
    });
});

test('A simple pre-phase expansion', function(t) {
    t.plan(7);
    expansions.expandString('<(a)', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, '1');
    });
    expansions.expandString('<(a)23', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, '123');
    });
    expansions.expandString('0<(a)2', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, '012');
    });
    expansions.expandString('<(a))', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, '1)');
    });
    expansions.expandString('(<(a)', { a: '1' }, 'pre', function (e, r) {
       t.equal(r, '(1');
    });
    expansions.expandString('<<(a)', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, '<1');
    });
    expansions.expandString('>(a)', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, '>(a)');
    });
});

test('A simple post-phase expansion', function(t) {
    t.plan(7);
    expansions.expandString('>(a)', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '1');
    });
    expansions.expandString('>(a)23', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '123');
    });
    expansions.expandString('0>(a)2', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '012');
    });
    expansions.expandString('>(a))', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '1)');
    });
    expansions.expandString('(>(a)', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '(1');
    });
    expansions.expandString('>>(a)', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '>1');
    });
    expansions.expandString('<(a)', { a: '1' }, 'post', function(e, r) {
        t.equal(r, '<(a)');
    });
});

test('Handle command execution', function(t) {
    t.plan(2);
    expansions.expandString('<!(echo hi)', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, 'hi');
    });
    expansions.expandString('<!(["echo","hi"])', { a: '1' }, 'pre', function (e, r) {
        t.equal(r, 'hi');
    });
});
