# to-utf8

given an integer-indexable input iterable, produce a utf8 string.

```javascript
  var utf8 = require('to-utf8')

  var buf = new Buffer('淾淾淾 hello world 淾淾 yep ƒuu 淾', 'utf8')
  assert.equal(utf8(buf), buf.toString('utf8'))
```

## api

#### utf8(iterable[, byteStart][, byteEnd]) -> string

take an iterable of bytes (`0...0xFF`) and produce an utf8 string.

# license

MIT
