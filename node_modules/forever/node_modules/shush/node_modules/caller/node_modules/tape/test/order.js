var test = require('../');
var current = 0;

test(function (t) {
    t.equal(current++, 0);
    t.end();
});
test(function (t) {
    t.plan(1);
    setTimeout(function () {
        t.equal(current++, 1);
    }, 100);
});
test(function (t) {
    t.equal(current++, 2);
    t.end();
});
