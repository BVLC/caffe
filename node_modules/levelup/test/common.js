/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var referee = require('referee')
  , assert  = referee.assert
  , refute  = referee.refute
  , crypto  = require('crypto')
  , async   = require('async')
  , rimraf  = require('rimraf')
  , fs      = require('fs')
  , path    = require('path')
  , levelup = require('../lib/levelup.js')
  , child_process = require('child_process')
  , dbidx   = 0

referee.add('isInstanceOf', {
    assert: function (actual, expected) {
        return actual instanceof expected
    }
  , refute: function (actual, expected) {
        return !(actual instanceof expected)
    }
  , assertMessage: '${0} expected to be instance of ${1}'
  , refuteMessage: '${0} expected not to be instance of ${1}'
})

referee.add('isUndefined', {
    assert: function (actual) {
        return actual === undefined
    }
  , refute: function (actual) {
        return actual !== undefined
    }
  , assertMessage: '${0} expected to be undefined'
  , refuteMessage: '${0} expected not to be undefined'
})

module.exports.nextLocation = function () {
  return path.join(__dirname, '_levelup_test_db_' + dbidx++)
}

module.exports.cleanup = function (callback) {
  fs.readdir(__dirname, function (err, list) {
    if (err) return callback(err)

    list = list.filter(function (f) {
      return (/^_levelup_test_db_/).test(f)
    })

    if (!list.length)
      return callback()

    var ret = 0

    list.forEach(function (f) {
      rimraf(path.join(__dirname, f), function () {
        if (++ret == list.length)
          callback()
      })
    })
  })
}

module.exports.openTestDatabase = function () {
  var options = typeof arguments[0] == 'object' ? arguments[0] : { createIfMissing: true, errorIfExists: true }
    , callback = typeof arguments[0] == 'function' ? arguments[0] : arguments[1]
    , location = typeof arguments[0] == 'string' ? arguments[0] : module.exports.nextLocation()

  rimraf(location, function (err) {
    refute(err)
    this.cleanupDirs.push(location)
    levelup(location, options, function (err, db) {
      refute(err)
      if (!err) {
        this.closeableDatabases.push(db)
        callback(db)
      }
    }.bind(this))
  }.bind(this))
}

module.exports.commonTearDown = function (done) {
  async.forEach(
      this.closeableDatabases
    , function (db, callback) {
        db.close(callback)
      }
    , module.exports.cleanup.bind(null, done)
  )
}

module.exports.loadBinaryTestData = function (callback) {
  fs.readFile(path.join(__dirname, 'data/testdata.bin'), callback)
}

module.exports.binaryTestDataMD5Sum = '920725ef1a3b32af40ccd0b78f4a62fd'

module.exports.checkBinaryTestData = function (testData, callback) {
  var md5sum = crypto.createHash('md5');
  md5sum.update(testData)
  assert.equals(md5sum.digest('hex'), module.exports.binaryTestDataMD5Sum)
  callback()
}

module.exports.commonSetUp = function (done) {
  this.cleanupDirs = []
  this.closeableDatabases = []
  this.openTestDatabase = module.exports.openTestDatabase.bind(this)
  this.timeout = 10000
  module.exports.cleanup(done)
}

module.exports.readStreamSetUp = function (done) {
  module.exports.commonSetUp.call(this, function () {
    var i, k

    this.dataSpy    = this.spy()
    this.endSpy     = this.spy()
    this.sourceData = []

    for (i = 0; i < 100; i++) {
      k = (i < 10 ? '0' : '') + i
      this.sourceData.push({
          type  : 'put'
        , key   : k
        , value : Math.random()
      })
    }

    this.verify = function (rs, done, data) {
      if (!data) data = this.sourceData // can pass alternative data array for verification
      assert.equals(this.endSpy.callCount, 1, 'ReadStream emitted single "end" event')
      assert.equals(this.dataSpy.callCount, data.length, 'ReadStream emitted correct number of "data" events')
      data.forEach(function (d, i) {
        var call = this.dataSpy.getCall(i)
        if (call) {
          //console.log('call', i, ':', call.args[0].key, '=', call.args[0].value, '(expected', d.key, '=', d.value, ')')
          assert.equals(call.args.length, 1, 'ReadStream "data" event #' + i + ' fired with 1 argument')
          refute.isNull(call.args[0].key, 'ReadStream "data" event #' + i + ' argument has "key" property')
          refute.isNull(call.args[0].value, 'ReadStream "data" event #' + i + ' argument has "value" property')
          assert.equals(call.args[0].key, d.key, 'ReadStream "data" event #' + i + ' argument has correct "key"')
          assert.equals(+call.args[0].value, +d.value, 'ReadStream "data" event #' + i + ' argument has correct "value"')
        }
      }.bind(this))
      done()
    }.bind(this)

    done()

  }.bind(this))
}
