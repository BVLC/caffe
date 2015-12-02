/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var assert       = require('referee').assert
  , fs           = require('fs')
  , path         = require('path')
  , fstream      = require('fstream')
  , tar          = require('tar')
  , crypto       = require('crypto')
  , levelup      = require('../../lib/levelup')

  , dblocation   = path.join(__dirname, 'levelup_test_binary.db')
  , datatar      = path.join(__dirname, 'test-data.tar')
  , datadir      = path.join(__dirname, 'test-data')
  , db
  , expectedEntries

module.exports.dblocation = dblocation
module.exports.datatar    = datatar
module.exports.datadir    = datadir

module.exports.opendb = function (dblocation, callback) {
  levelup(
      dblocation
    , { createIfMissing: true , errorIfExists: false, keyEncoding: 'utf8', valueEncoding: 'binary' }
    , function (err, _db) {
        db = _db
        console.log('Opened database...')
        callback(err)
      }
  )
}

module.exports.extract = function (tarfile, dir, callback) {
  expectedEntries = 0
  fs.createReadStream(tarfile)
    .pipe(tar.Extract({ path: dir }))
    .on('entry', function (entry) {
      if (entry.props.File || entry.File || entry.type == 'File')
        expectedEntries++
    })
    .on('end', function () {
      console.log('Extracted tar file...')
      callback()
    })
}

module.exports.fstreamWrite = function (callback) {
  fstream.Reader(datadir)
    .pipe(db.writeStream({ fstreamRoot: path.resolve(__dirname) })
      .on('close', function () {
        console.log('Piped data to database...')
        callback()
      }))
      .on('error', callback)
}

// using sync:true will force a flush to the fs, otherwise the readStream() is too
// quick and won't get the full data
module.exports.sync = function (callback) {
  db.put('__', '__', { sync: true }, function (err) {
    if (err) return callback(err)
    db.del('__', { sync: true }, callback)
  })
}

module.exports.verify = function (callback) {
  var entries = 0
  db.readStream()
    .on('data', function (data) {
      var md5sum = crypto.createHash('md5')
        , dbmd5sum

      md5sum.update(data.value)
      dbmd5sum = md5sum.digest('hex')
      md5sum = crypto.createHash('md5')
      entries++
      fs.createReadStream(path.join(__dirname, data.key))
        .on('data', function (d) { md5sum.update(d) })
        .on('end', function () {
          var fsmd5sum = md5sum.digest('hex')
          assert.equals(
              dbmd5sum
            , fsmd5sum
            , 'MD5 sum compare of ' + data.key + ' failed (' + dbmd5sum + ' != ' + fsmd5sum + ')'
          )
        })
    })
    .on('end', function () {
      assert.equals(entries, expectedEntries, 'correct number of entries in the database')
      console.log('Finished comparing database entries...')
      console.log('Cleaning up...')
      callback()
    })
}