var test = require('../');

test('parent', function (t) {
    t.plan(3)

    var firstChildRan = false;

    t.pass('assertion in parent');

    t.test('first child', function (t) {
        t.plan(1);
        t.pass('pass first child');
        firstChildRan = true;
    });

    t.test('second child', function (t) {
        t.plan(2);
        t.ok(firstChildRan, 'first child ran first');
        t.pass('pass second child');
    });
});
