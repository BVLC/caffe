var test = require('../');

test('only3 test 1', function (t) {
    t.fail('not 1');
    t.end();
});

test.only('only3 test 2', function (t) {
    t.end();
});

test('only3 test 3', function (t) {
    t.fail('not 3');
    t.end();
});
