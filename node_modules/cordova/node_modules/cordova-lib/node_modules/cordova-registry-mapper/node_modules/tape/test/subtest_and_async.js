var test = require('../');

var asyncFunction = function (callback) {
  setTimeout(callback, Math.random * 50);
};

test('master test', function (t) {
  t.test('subtest 1', function (t) {
    t.pass('subtest 1 before async call');
    asyncFunction(function () {
      t.pass('subtest 1 in async callback');
      t.end();
    })
  });

  t.test('subtest 2', function (t) {
    t.pass('subtest 2 before async call');
    asyncFunction(function () {
      t.pass('subtest 2 in async callback');
      t.end();
    })
  });
});
