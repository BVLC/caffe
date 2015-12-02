var test = require('../');
var ran = 0;

test('do not skip this', { skip: false }, function(t) {
    t.pass('this should run');
    ran ++;
    t.end();
});

test('skip this', { skip: true }, function(t) {
    t.fail('this should not even run');
    t.end();
});

test('skip subtest', function(t) {
    ran ++;
    t.test('do not skip this', { skip: false }, function(t) {
        ran ++;
        t.pass('this should run');
        t.end();
    });
    t.test('skip this', { skip: true }, function(t) {
        t.fail('this should not even run');
        t.end();
    });
    t.end();
});

test('right number of tests ran', function(t) {
    t.equal(ran, 3, 'ran the right number of tests');
    t.end();
});

// vim: set softtabstop=4 shiftwidth=4:
