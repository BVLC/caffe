var test = require('../');

test('plan should be optional', function (t) {
    t.pass('no plan here');
    t.end();
});

test('no plan async', function (t) {
    setTimeout(function() {
        t.pass('ok');
        t.end();
    }, 100);
});

// vim: set softtabstop=4 shiftwidth=4:
