/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var async      = require('async')
  , du         = require('du')
  , delayed    = require('delayed')
  , levelup    = require('../')
  , common     = require('./common')

  , assert  = require('referee').assert
  , refute  = require('referee').refute
  , buster  = require('bustermove')

  , compressableData = new Buffer(Array.apply(null, Array(1024 * 100)).map(function () { return 'aaaaaaaaaa' }).join(''))
  , multiples = 10
  , dataSize = compressableData.length * multiples

  , verify = function (location, compression, done) {
      du(location, function (err, size) {
        if (err) return refute(err)
        //console.log(Math.round((size / dataSize) * 100) + '% compression ratio (', size, 'b vs', dataSize, 'b)')
        if (compression)
          assert(size < dataSize, 'on-disk size (' + size + ') is less than data size (' + dataSize + ')')
        else
          assert(size >= dataSize, 'on-disk size (' + size + ') is greater than data size (' + dataSize + ')')
        done()
      })
    }

    // close, open, close again.. 'compaction' is also performed on open()s
  , cycle = function (db, compression, callback) {
      var location = db.location
      db.close(function (err) {
        if (err) return refute(err)
        levelup(location, { errorIfExists: false, compression: compression }, function (err, db) {
          if (err) return refute(err)
          db.close(function (err) {
            if (err) return refute(err)
            callback()
          })
        })
      })
    }

buster.testCase('Compression', {
    'setUp': common.readStreamSetUp

  , 'tearDown': common.commonTearDown

  , 'test data is compressed by default (db.put())': function (done) {
      this.openTestDatabase(function (db) {
        async.forEach(
            Array.apply(null, Array(multiples)).map(function (e, i) {
              return [ i, compressableData ]
            })
          , function (args, callback) {
              db.put.apply(db, args.concat([callback]))
            }
          , cycle.bind(null, db, true, delayed.delayed(verify.bind(null, db.location, true, done), 0.01))
        )
      })
    }

  , 'test data is not compressed with compression=false on open() (db.put())': function (done) {
      this.openTestDatabase({ createIfMissing: true, errorIfExists: true, compression: false }, function (db) {
        async.forEach(
            Array.apply(null, Array(multiples)).map(function (e, i) {
              return [ i, compressableData ]
            })
          , function (args, callback) {
              db.put.apply(db, args.concat([callback]))
            }
          , cycle.bind(null, db, false, delayed.delayed(verify.bind(null, db.location, false, done), 0.01))
        )
      })
    }

  , 'test data is compressed by default (db.batch())': function (done) {
      this.openTestDatabase(function (db) {
        db.batch(
            Array.apply(null, Array(multiples)).map(function (e, i) {
              return { type: 'put', key: i, value: compressableData }
            })
          , cycle.bind(null, db, false, delayed.delayed(verify.bind(null, db.location, false, done), 0.01))
        )
      })
    }
})