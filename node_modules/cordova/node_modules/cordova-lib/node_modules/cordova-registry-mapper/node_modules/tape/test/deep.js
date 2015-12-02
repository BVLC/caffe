var test = require('../');

test('deep strict equal', function (t) {
    t.notDeepEqual(
        [ { a: '3' } ],
        [ { a: 3 } ]
    );
    t.end();
});
