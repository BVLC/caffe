var test = require('../')

test('parent', function (t) {
  t.pass('parent');
  setTimeout(function () {
    t.test('child', function (t) {
      t.pass('child');
      t.end();
    });
  }, 100)
})
