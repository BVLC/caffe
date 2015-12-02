var test = require('tape')
  , toUTF8 = require('./index')

test('works as expected', function(assert) {
  var input = ['11000010', '10100010'].map(function(x) { return parseInt(x, 2) })
  assert.equal(toUTF8(input), '\u00a2')
  assert.end() 
})

test('works the same as buffer.toString(utf8)', function(assert) {
  var buf = new Buffer('淾淾淾 hello world 淾淾 yep ƒuu 淾', 'utf8')
  assert.equal(toUTF8(buf), buf.toString('utf8'))
  assert.end()
})
