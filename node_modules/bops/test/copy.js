var test = require('tape')
  , binary = require('../index')

test('copy works as expected', function(assert) {
  var tmp1 = binary.create(16)
    , tmp2 = binary.create(16)

  for(var i = 0, len = 16; i < len; ++i) {
    binary.writeUInt8(tmp1, Math.random() * 0xFF & 0xFF, i)
    binary.writeUInt8(tmp2, Math.random() * 0xFF & 0xFF, i) 
  }

  binary.copy(tmp1, tmp2)
  for(var i = 0, len = 16; i < len; ++i) {
    assert.equal(binary.readUInt8(tmp2, i), binary.readUInt8(tmp1, i))
  }

  // overlapping copy
  binary.copy(tmp1, tmp1, 2, 0, 6)

  for(var i = 2, len = 8; i < len; ++i) {
    assert.equal(binary.readUInt8(tmp1, i), binary.readUInt8(tmp2, i - 2)) 
  }
  assert.end()
})
