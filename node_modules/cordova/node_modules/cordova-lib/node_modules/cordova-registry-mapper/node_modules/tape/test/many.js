var test = require('../');

test('many tests', function (t) {
    t.plan(100);
    for (var i = 0; i < 100; i++) {
        setTimeout(function () { t.pass() }, Math.random() * 50);
    }
});
