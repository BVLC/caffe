var test = require('../../../');
test(function (t) {
    t.plan(2);
    t.equal(1+1, 2);
    t.ok(true);
});

test('wheee', function (t) {
    t.ok(true);
    t.end();
});
