var test = require('../');

test('one', function (t) {
    t.plan(2);
    t.ok(true);
    setTimeout(function () {
        t.equal(1+3, 4);
    }, 100);
});

test('two', function (t) {
    t.plan(3);
    t.equal(5, 2+3);
    setTimeout(function () {
        t.equal('a'.charCodeAt(0), 97);
        t.ok(true);
    }, 50);
});
