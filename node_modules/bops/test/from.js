var test = require('tape')
  , binary = require('../index')

test('from array works', function(assert) {
  var arr = [1, 2, 3]
    , buf = binary.from(arr)

  assert.equal(buf.length, arr.length)
  for(var i = 0, len = arr.length; i < len; ++i) {
    assert.equal(binary.readUInt8(buf, i), arr[i])
  }
  assert.end()
})

test('from utf8 works as expected', function(assert) {
  var buf = binary.from('ƒello 淾淾淾 hello world 淾淾 yep ƒuu 淾', 'utf8')
    , expect

  expect = [198,146,101,108,108,111,32,230,183,190,230,183,190,230,183,190,32,104,101,108,108,111,32,119,111,114,108,100,32,230,183,190,230,183,190,32,121,101,112,32,198,146,117,117,32,230,183,190]

  assert.equal(buf.length, expect.length)
  for(var i = 0, len = buf.length; i < len; ++i) {
    assert.equal(binary.readUInt8(buf, i), expect[i])
  }

  assert.end()
})

test('from hex works as expected', function(assert) {
  var buf = binary.from('68656c6c6f20776f726c64c692000a0809', 'hex')
    , expect

  expect = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 198, 146, 0, 10, 8, 9]

  assert.equal(buf.length, expect.length)
  for(var i = 0, len = buf.length; i < len; ++i) {
    assert.equal(binary.readUInt8(buf, i), expect[i])
  }

  assert.end()
})

test('from base64 works as expected', function(assert) {
  var buf = binary.from('aGVsbG8gd29ybGTGkgAKCAk=', 'base64')
    , expect

  expect = [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 198, 146, 0, 10, 8, 9]

  assert.equal(buf.length, expect.length)
  for(var i = 0, len = buf.length; i < len; ++i) {
    assert.equal(binary.readUInt8(buf, i), expect[i])
  }

  assert.end()
})
