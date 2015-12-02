var test = require('tape')
  , binary = require('../index')

test('to utf8 works as expected', function(assert) {
  var buf = binary.from([198,146,101,108,108,111,32,230,183,190,230,183,190,230,183,190,32,104,101,108,108,111,32,119,111,114,108,100,32,230,183,190,230,183,190,32,121,101,112,32,198,146,117,117,32,230,183,190])
    , expect = 'ƒello 淾淾淾 hello world 淾淾 yep ƒuu 淾'

  assert.equal(expect, binary.to(buf, 'utf8'))
  assert.end()
})

test('from hex works as expected', function(assert) {
  var buf = binary.from([104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 198, 146, 0, 10, 8, 9])
    , expect = '68656c6c6f20776f726c64c692000a0809'

  assert.equal(binary.to(buf, 'hex'), expect)
  assert.end()
})

test('from base64 works as expected', function(assert) {
  var buf = binary.from([104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 198, 146, 0, 10, 8, 9])
    , expect = 'aGVsbG8gd29ybGTGkgAKCAk='

  assert.equal(binary.to(buf, 'base64'), expect)
  assert.end()
})
