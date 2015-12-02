/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

/*
 * This test verifies that an existing database contains the
 * correct data, by comparing it to the original data contained
 * in a tar file.
 * Useful for comparing across LevelDB versions.
 */

var async      = require('async')
  , rimraf     = require('rimraf')
  , path       = require('path')
  , tarcommon  = require('./tarcommon')

  , dbtar      = path.join(__dirname, 'test-data.db.tar')
  , dblocation = path.join(__dirname, 'levelup_test_compat.db')

function runTest (dbtar, callback) {
  async.series([
      // pre-clean
      rimraf.bind(null, tarcommon.dblocation)
    , rimraf.bind(null, dblocation)
    , rimraf.bind(null, tarcommon.datadir)
      // extract existing database
    , tarcommon.extract.bind(null, dbtar, __dirname)
      // extract data for comparison
    , tarcommon.extract.bind(null, tarcommon.datatar, tarcommon.datadir)
      // open database
    , tarcommon.opendb.bind(null, dblocation)
      // verify database entries are the same as the files
    , tarcommon.verify
      // clean up
    , rimraf.bind(null, tarcommon.dblocation)
    , rimraf.bind(null, dblocation)
    , rimraf.bind(null, tarcommon.datadir)
  ], callback)
}

console.log('***************************************************')
console.log('RUNNING COMPAT-DATA-TEST...')

runTest(dbtar, function (err) {
  if (err) throw err
  console.log('No errors? All good then!')
  console.log('***************************************************')
  process.exit(err ? -1 : 0)
})