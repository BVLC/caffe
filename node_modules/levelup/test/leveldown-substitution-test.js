/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var levelup = require('../lib/levelup.js')
  , assert  = require('referee').assert
  , refute  = require('referee').refute
  , buster  = require('bustermove')
  , MemDOWN = require('memdown')

require('./common')

buster.testCase('LevelDOWN Substitution', {
    'test substitution of LevelDOWN with MemDOWN': function (done) {
      var md       = new MemDOWN('foo')
        , db       =
            levelup('/somewhere/not/writable/booya!', { db: function () { return md } })
        , entries  = []
        , expected = [
              { key: 'a', value: 'A' }
            , { key: 'b', value: 'B' }
            , { key: 'c', value: 'C' }
            , { key: 'd', value: 'D' }
            , { key: 'e', value: 'E' }
            , { key: 'f', value: 'F' }
            , { key: 'i', value: 'I' }
          ]

      db.put('f', 'F')
      db.put('h', 'H')
      db.put('i', 'I')
      db.put('a', 'A')
      db.put('c', 'C')
      db.put('e', 'E')
      db.del('g')
      db.batch([
          { type: 'put', key: 'd', value: 'D' }
        , { type: 'del', key: 'h' }
        , { type: 'put', key: 'b', value: 'B' }
      ])

      db.createReadStream()
        .on('data', function (data) { entries.push(data) })
        .on('error', function (err) { refute(err, 'readStream emitted an error') })
        .on('close', function () {
          assert.equals(entries, expected, 'correct entries')
          assert.equals(
              md._keys
            , expected.map(function (e) { return e.key })
            , 'memdown has the entries'
          )
          done()
        })
    }
})
