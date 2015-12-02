var test = require('../../');

test(function (t) {
    t.plan(4);
    t.ok(true);
    t.equal(3, 1+2);
    t.deepEqual([1,2,[3,4]], [1,2,[3,4]]);
    t.notDeepEqual([1,2,[3,4,5]], [1,2,[3,4]]);
});
