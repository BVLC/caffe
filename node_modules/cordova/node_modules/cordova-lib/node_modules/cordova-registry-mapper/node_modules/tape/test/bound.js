var test = require('../');

test('bind works', function (t) {
  t.plan(2);
  var equal = t.equal;
  var deepEqual = t.deepEqual;
  equal(3, 3);
  deepEqual([4], [4]);
  t.end();
});
