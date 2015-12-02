var inspect = require('../');
var test = require('tape');

test('function', function (t) {
    t.plan(1);
    var obj = [ 1, 2, function f (n) { return n + 5 }, 4 ];
    t.equal(inspect(obj), '[ 1, 2, [Function: f], 4 ]');
});
