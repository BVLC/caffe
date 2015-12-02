/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var levelup = require('../lib/levelup.js')
  , errors  = require('../lib/errors.js')
  , common  = require('./common')

  , assert  = require('referee').assert
  , refute  = require('referee').refute
  , buster  = require('bustermove')

buster.testCase('null & undefined keys & values', {
    'setUp': common.commonSetUp
  , 'tearDown': common.commonTearDown

  , 'null and undefined': {
        'setUp': function (done) {
          levelup(this.cleanupDirs[0] = common.nextLocation(), { createIfMissing: true }, function (err, db) {
            refute(err) // sanity
            this.closeableDatabases.push(db)
            assert.isTrue(db.isOpen())
            this.db = db
            done()
          }.bind(this))
        }

      , 'get() with null key causes error': function (done) {
          this.db.get(null, function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'get() with undefined key causes error': function (done) {
          this.db.get(undefined, function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'del() with null key causes error': function (done) {
          this.db.del(null, function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'del() with undefined key causes error': function (done) {
          this.db.del(undefined, function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'put() with null key causes error': function (done) {
          this.db.put(null, 'foo', function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'put() with undefined key causes error': function (done) {
          this.db.put(undefined, 'foo', function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'put() with null value causes error': function (done) {
          this.db.put('foo', null, function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }

      , 'put() with undefined value causes error': function (done) {
          this.db.put('foo', undefined, function (err, value) {
            refute(value)
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }
      , 'batch() with undefined value causes error': function (done) {
          this.db.batch([{key: 'foo', value: undefined, type: 'put'}]
          , function (err) {
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }
      , 'batch() with null value causes error': function (done) {
          this.db.batch([{key: 'foo', value: null, type: 'put'}]
          , function (err) {
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }
      , 'batch() with undefined key causes error': function (done) {
          this.db.batch([{key: undefined, value: 'bar', type: 'put'}]
          , function (err) {
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }
      , 'batch() with null key causes error': function (done) {
          this.db.batch([{key: null, value: 'bar', type: 'put'}]
          , function (err) {
            assert.isInstanceOf(err, Error)
            assert.isInstanceOf(err, errors.LevelUPError)
            done()
          })
        }
    }
})
