var Walker = require('./walker')

function walk (path, filter) {
  return new Walker(path, filter).start()
}

module.exports = walk
