module.exports = create

var Buffer = require('buffer').Buffer

function create(size) {
  return new Buffer(size)
}
