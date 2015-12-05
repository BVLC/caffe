
/**
 * Decode the given `buf`.
 *
 * @param {Buffer} buf
 * @return {Object}
 * @api public
 */

module.exports = function(buf){
  var off = 0;

  // unpack meta
  var meta = buf[off++];
  var version = meta >> 4;
  var argv = meta & 0xf;
  var args = new Array(argv);

  // unpack args
  for (var i = 0; i < argv; i++) {
    var len = buf.readUInt32BE(off);
    off += 4;

    var arg = buf.slice(off, off += len);
    args[i] = arg;
  }

  return args;
};