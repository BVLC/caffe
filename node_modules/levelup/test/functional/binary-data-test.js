/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

/*
 * This test unpacks a tar file, pushes that data into a
 * database then compares the database data with the files
 * on the filesystem.
 * The different types of data are useful for testing, particularly
 * the binary files.
 */

var async     = require('async')
  , rimraf    = require('rimraf')
  , tarcommon = require('./tarcommon')

console.log('***************************************************')
console.log('RUNNING BINARY-DATA-TEST...')

async.series([
    // pre-clean
    rimraf.bind(null, tarcommon.dblocation)
  , rimraf.bind(null, tarcommon.datadir)
    // extract data for comparison
  , tarcommon.extract.bind(null, tarcommon.datatar, tarcommon.datadir)
    // open database
  , tarcommon.opendb.bind(null, tarcommon.dblocation)
    // push the data into a database
  , tarcommon.fstreamWrite
    // run a sync put & del to force an fs sync
  , tarcommon.sync
    // verify database entries are the same as the files
  , tarcommon.verify
    // clean up
  , rimraf.bind(null, tarcommon.dblocation)
  , rimraf.bind(null, tarcommon.datadir)
], function (err) {
  if (err) console.error('Error', err)
  else console.log('No errors? All good then!')
  console.log('***************************************************')
  process.exit(err ? -1 : 0)
})