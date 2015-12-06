(function() {
  var e, indexOf, isBuffer, max, min,
    __slice = [].slice;

  isBuffer = Buffer.isBuffer;

  min = Math.min, max = Math.max;

  try {
    exports.indexOf = require('buffertools').indexOf;
  } catch (_error) {
    e = _error;
    require('bufferjs/indexOf');
    indexOf = Buffer.indexOf;
    exports.indexOf = function(needle) {
      return indexOf(this, needle);
    };
    exports.warn = "Warning: using slow naiv Buffer.indexOf function!\n" + "`npm install buffertools` to speed things up.";
    process.nextTick(function() {
      if (exports.warn) {
        return console.warn(exports.warn);
      }
    });
  }

  if (Buffer.concat != null) {
    exports.concat = (function() {
      var args;
      args = 1 <= arguments.length ? __slice.call(arguments, 0) : [];
      return Buffer.concat(args);
    });
  }

  if (exports.concat == null) {
    exports.concat = function() {
      var args, buffer, buffers, i, idx, input, length, pos, result, _i, _j, _len, _len1;
      args = 1 <= arguments.length ? __slice.call(arguments, 0) : [];
      idx = -1;
      length = 0;
      buffers = [];
      for (i = _i = 0, _len = args.length; _i < _len; i = ++_i) {
        input = args[i];
        if (isBuffer(input)) {
          if (input.length) {
            idx = i;
          }
          length += input.length;
          buffers.push(input);
        }
      }
      if (idx !== -1 && length === args[idx].length) {
        return args[idx];
      }
      pos = 0;
      result = new Buffer(length);
      for (_j = 0, _len1 = buffers.length; _j < _len1; _j++) {
        buffer = buffers[_j];
        if (!buffer.length) {
          continue;
        }
        buffer.copy(result, pos);
        pos += buffer.length;
      }
      return result;
    };
  }

  exports.split = function(buffer, pos, offset) {
    var buflen, found, rest;
    if (offset == null) {
      offset = 0;
    }
    buflen = buffer.length;
    found = new Buffer(min(buflen, pos));
    rest = new Buffer(max(0, buflen - pos - offset));
    buffer.copy(found, 0, 0, min(buflen, pos));
    buffer.copy(rest, 0, min(buflen, pos + offset));
    return [found, rest];
  };

}).call(this);
