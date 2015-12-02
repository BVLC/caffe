var test = require('tape')
  , binary = require('../index')

test('write works as expected', function(assert) {
  var tests = {
      writeUInt8:      [binary.from([16]), 16]
    , writeInt8:       [binary.from([16]), 16]
    , writeInt8:       [binary.from([0xff]), -1]
    , writeInt8:       [binary.from([0x80]), -128]
    , writeUInt16LE:   [binary.from([0xCF, 0xAF]), 45007]
    , writeUInt32LE:   [binary.from([0xAF, 0, 0, 0xCF]), 3472883887]
    , writeInt16LE:    [binary.from([0xCF, 0xAF]), -20529]
    , writeInt32LE:    [binary.from([0xCF, 0, 0, 0xCF]), -822083377]
    , writeFloatLE:    [binary.from([0xCF, 0xAF, 0xDA, 0x02]), 3.213313024388152e-37]
    , writeDoubleLE:   [binary.from([0xcf, 0xaf, 0xda, 0x02, 0x00, 0x01, 0x2f, 0x44])
                      , 285960563508654870000]
    , writeUInt16BE:   [binary.from([0xCF, 0xAF]), 53167]
    , writeUInt32BE:   [binary.from([0xAF, 0, 0, 0xCF]), 2936013007]
    , writeInt16BE:    [binary.from([0xCF, 0xAF]), -12369]
    , writeInt32BE:    [binary.from([0xAF, 0, 0, 0xCF]), -1358954289]
    , writeFloatBE:    [binary.from([0xCF, 0xAF, 0xDA, 0x02]), -5900600320]
    , writeDoubleBE:   [binary.from([0xcf, 0xaf, 0xda, 0x02, 0x00, 0x01, 0x2f, 0x44]), -7.203442384910198e+75]
  }

  for(var key in tests) {
    var expect = tests[key][0]
      , value = tests[key][1]
      , buf = binary.create(expect.length)

    binary[key](buf, value, 0)
    for(var i = 0, len = expect.length; i < len; ++i) {
      assert.equal(
        binary.readUInt8(buf, i)
      , binary.readUInt8(expect, i)
      )
    }
  }
  assert.end()
})

