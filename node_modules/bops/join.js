var Buffer = require('buffer').Buffer

module.exports = function(targets, hint) {
  return hint !== undefined ?
    Buffer.concat(targets, hint) :
    Buffer.concat(targets)
}
