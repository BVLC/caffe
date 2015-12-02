var test = require('tape')
  , binary = require('../index')

test('read works as expected', function(assert) {
  var tests = {
      readUInt8:      [binary.from([16]), 16]
    , readInt8:       [binary.from([16]), 16]
    , readInt8:       [binary.from([0xff]), -1]
    , readInt8:       [binary.from([0x80]), -128]
    , readUInt16LE:   [binary.from([0xCF, 0xAF]), 45007]
    , readUInt32LE:   [binary.from([0xAF, 0, 0, 0xCF]), 3472883887]
    , readInt16LE:    [binary.from([0xCF, 0xAF]), -20529]
    , readInt32LE:    [binary.from([0xCF, 0, 0, 0xCF]), -822083377]
    , readFloatLE:    [binary.from([0xCF, 0xAF, 0xDA, 0x02]), 3.213313024388152e-37]
    , readDoubleLE:   [binary.from([0xcf, 0xaf, 0xda, 0x02, 0x00, 0x01, 0x2f, 0x44])
                      , 285960563508654870000]
    , readUInt16BE:   [binary.from([0xCF, 0xAF]), 53167]
    , readUInt32BE:   [binary.from([0xAF, 0, 0, 0xCF]), 2936013007]
    , readInt16BE:    [binary.from([0xCF, 0xAF]), -12369]
    , readInt32BE:    [binary.from([0xAF, 0, 0, 0xCF]), -1358954289]
    , readFloatBE:    [binary.from([0xCF, 0xAF, 0xDA, 0x02]), -5900600320]
    , readDoubleBE:   [binary.from([0xcf, 0xaf, 0xda, 0x02, 0x00, 0x01, 0x2f, 0x44]), -7.203442384910198e+75]
  }

  for(var key in tests) {
    var buf = tests[key][0]
      , expect = tests[key][1]

    assert.equal(binary[key](buf, 0), expect)
  }
  assert.end()
})
