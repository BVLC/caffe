var test = require('tape')
  , binary = require('../index')

test('subarray works as expected', function(assert) {
  var tmp1 = binary.create(16)
    , tmp2

  for(var i = 0, len = 16; i < len; ++i) {
    binary.writeUInt8(tmp1, Math.random() * 0xFF & 0xFF, i)
  }

  tmp2 = binary.subarray(tmp1, 4)
  binary.writeUInt8(tmp2, 255, 0)

  assert.equal(binary.readUInt8(tmp1, 4), 255)
  assert.end()
})
