if (process.env.OBJECT_IMPL) global.TYPED_ARRAY_SUPPORT = false
var B = require('../').Buffer
var test = require('tape')

test('buf.constructor is Buffer', function (t) {
  var buf = new B([1, 2])
  t.strictEqual(buf.constructor, B)
  t.end()
})

test('instanceof Buffer', function (t) {
  var buf = new B([1, 2])
  t.ok(buf instanceof B)
  t.end()
})

test('convert to Uint8Array in modern browsers', function (t) {
  if (B.TYPED_ARRAY_SUPPORT) {
    var buf = new B([1, 2])
    var uint8array = new Uint8Array(buf.buffer)
    t.ok(uint8array instanceof Uint8Array)
    t.equal(uint8array[0], 1)
    t.equal(uint8array[1], 2)
  } else {
    t.pass('object impl: skipping test')
  }
  t.end()
})

test('indexes from a string', function (t) {
  var buf = new B('abc')
  t.equal(buf[0], 97)
  t.equal(buf[1], 98)
  t.equal(buf[2], 99)
  t.end()
})

test('indexes from an array', function (t) {
  var buf = new B([ 97, 98, 99 ])
  t.equal(buf[0], 97)
  t.equal(buf[1], 98)
  t.equal(buf[2], 99)
  t.end()
})

test('setting index value should modify buffer contents', function (t) {
  var buf = new B([ 97, 98, 99 ])
  t.equal(buf[2], 99)
  t.equal(buf.toString(), 'abc')

  buf[2] += 10
  t.equal(buf[2], 109)
  t.equal(buf.toString(), 'abm')
  t.end()
})

test('storing negative number should cast to unsigned', function (t) {
  var buf = new B(1)

  if (B.TYPED_ARRAY_SUPPORT) {
    // This does not work with the object implementation -- nothing we can do!
    buf[0] = -3
    t.equal(buf[0], 253)
  }

  buf = new B(1)
  buf.writeInt8(-3, 0)
  t.equal(buf[0], 253)

  t.end()
})

// TODO: test write negative with
