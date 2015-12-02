'use strict';
if (process.env.OBJECT_IMPL) global.TYPED_ARRAY_SUPPORT = false;
var Buffer = require('../../').Buffer;

var common = {};
var assert = require('assert');

var Buffer = require('../../').Buffer;
var LENGTH = 16;

var ab = new ArrayBuffer(LENGTH);
var dv = new DataView(ab);
var ui = new Uint8Array(ab);
var buf = new Buffer(ab);


assert.ok(Buffer.isBuffer(buf));
// For backwards compatibility of old .parent property test that if buf is not
// a slice then .parent should be undefined.
assert.equal(buf.parent, undefined);
assert.equal(buf.buffer, ab);
assert.equal(buf.length, ab.byteLength);


buf.fill(0xC);
for (var i = 0; i < LENGTH; i++) {
  assert.equal(ui[i], 0xC);
  ui[i] = 0xF;
  assert.equal(buf[i], 0xF);
}

buf.writeUInt32LE(0xF00, 0);
buf.writeUInt32BE(0xB47, 4);
buf.writeDoubleLE(3.1415, 8);

assert.equal(dv.getUint32(0, true), 0xF00);
assert.equal(dv.getUint32(4), 0xB47);
assert.equal(dv.getFloat64(8, true), 3.1415);


// Now test protecting users from doing stupid things

assert.throws(function() {
  function AB() { }
  AB.__proto__ = ArrayBuffer;
  AB.prototype.__proto__ = ArrayBuffer.prototype;
  new Buffer(new AB());
}, TypeError);

