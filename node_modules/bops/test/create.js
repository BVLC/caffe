var test = require('tape')
  , binary = require('../index')

test('create works as expected', function(assert) {
  var len = Math.random() * 0xFF & 0xFF
  len += 1
  assert.equal(binary.create(len).length, len)
  assert.end()
})
