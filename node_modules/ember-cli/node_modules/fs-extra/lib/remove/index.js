var rimraf = require('rimraf')

function removeSync (dir) {
  return rimraf.sync(dir)
}

function remove (dir, callback) {
  return callback ? rimraf(dir, callback) : rimraf(dir, Function())
}

module.exports = {
  remove: remove,
  removeSync: removeSync,
  // alias
  delete: remove,
  deleteSync: removeSync
}
