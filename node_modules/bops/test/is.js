var test = require('tape')
  , binary = require('../index')

test('is works', function(assert) {
  var yes = binary.from("Hello")
    , no = "World"
    , never = { weird: true }
    , noway = [ 42 ]
  
  assert.equal(binary.is(yes), true)
  assert.equal(binary.is(no), false)
  assert.equal(binary.is(never), false)
  assert.equal(binary.is(noway), false)
  assert.end()
})
