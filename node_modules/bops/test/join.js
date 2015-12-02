var test = require('tape')
  , binary = require('../index')

test('test that join works as expected', function(assert) {
  var tmp1 = binary.create(16)
    , tmp2 = binary.create(16)
    , tmp3
    , cur

  for(var i = 0, len = 16; i < len; ++i) {
    binary.writeUInt8(tmp1, Math.random() * 0xFF & 0xFF, i)
    binary.writeUInt8(tmp2, Math.random() * 0xFF & 0xFF, i) 
  }

  tmp3 = binary.join([tmp1, tmp2])
  cur = tmp1

  for(var i = 0, j = 0, len = 32; i < len; ++i, ++j) {
    if(j !== 0 && j % 16 === 0) {
      cur = tmp2
      j = 0
    }
    assert.equal(
      binary.readUInt8(tmp3, i)
    , binary.readUInt8(cur, j)
    )
  }

  assert.end()
})
