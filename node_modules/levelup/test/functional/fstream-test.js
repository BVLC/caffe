/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var assert       = require('referee').assert
  , refute       = require('referee').refute
  , fstream      = require('fstream')
  , async        = require('async')
  , mkfiletree   = require('mkfiletree')
  , readfiletree = require('readfiletree')
  , rimraf       = require('rimraf')
  , bogan        = require('boganipsum')
  , levelup      = require('../../lib/levelup')

  , fixtureFiles = {
        'foo': 'FOO!\n'
      , 'a directory': {
            'bogantastic.txt': bogan()
          , 'subdir': {
                'boganmeup.dat': bogan()
              , 'sub sub dir': {
                    'bar': 'BAR!\n'
                  , 'maaaaaaaate': bogan()
                }
              , 'bang': 'POW'
            }
          , 'boo': 'W00t'
        }
    }
  , dblocation = 'levelup_test_fstream.db'

  , opendb = function (dir, callback) {
      levelup(dblocation, { createIfMissing: true , errorIfExists: false }, function (err, db) {
        refute(err)
        callback(null, dir, db)
      })
    }

  , fstreamWrite = function (dir, db, callback) {
      fstream.Reader(dir)
        .pipe(db.writeStream({ fstreamRoot: dir })
          .on('close', function () {
            db.close(function (err) {
              refute(err)
              callback(null, dir)
            })
          }))
    }

  , fstreamRead = function (dir, db, callback) {
      db.readStream({ type: 'fstream' })
        .pipe(new fstream.Writer({ path: dir + '.out', type: 'Directory' })
          .on('close', function () {
            db.close(function (err) {
              refute(err)
              callback(null, dir)
            })
          })
        )
    }

  , verify = function (dir, obj, callback) {
      assert.equals(obj, fixtureFiles)
      console.log('Guess what?? It worked!!')
      callback(null, dir)
    }

  , cleanUp = function (dir, callback) {
      async.parallel([
          rimraf.bind(null, dir + '.out')
        , rimraf.bind(null, dblocation)
        , mkfiletree.cleanUp
      ], callback)
    }

process.on('uncaughtException', function (err) {
  refute(err)
})

console.log('***************************************************')
console.log('RUNNING FSTREAM-TEST...')

async.waterfall([
    rimraf.bind(null, dblocation)
  , mkfiletree.makeTemp.bind(null, 'levelup_test_fstream', fixtureFiles)
  , opendb
  , fstreamWrite
  , opendb
  , fstreamRead
  , function (dir, callback) {
      readfiletree(dir, function (err, obj) {
        refute(err)
        callback(err, dir, obj)
      })
    }
  , verify
  , cleanUp
  , function () {
      console.log('***************************************************')
    }
])