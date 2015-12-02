var test = require('../../');

test('double end', function (t) {
    t.equal(1 + 1, 2);
    t.end();
    setTimeout(function () {
        t.end();
    }, 5);
});
