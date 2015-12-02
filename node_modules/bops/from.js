var Buffer = require('buffer').Buffer

module.exports = function(source, encoding) {
  return new Buffer(source, encoding)
}
