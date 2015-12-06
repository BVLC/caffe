(function() {
  var BufferStream, createReadStream, isBuffer, path, readFileSync, _ref;

  path = require('path');

  _ref = require('fs'), createReadStream = _ref.createReadStream, readFileSync = _ref.readFileSync;

  BufferStream = require('../');

  isBuffer = Buffer.isBuffer;

  module.exports = {
    defaults: function(æ) {
      var buffer, result, results, _i, _len, _ref1;
      buffer = new BufferStream({
        size: 'flexible'
      });
      æ.equal(buffer.length, 0);
      results = ["123", "bufferstream", "a", "bc", "def"];
      buffer.on('data', function(data) {
        return æ.equal(data.toString(), results.join(""));
      });
      buffer.on('end', function() {
        var buf;
        æ.equal(buffer.toString(), "");
        buf = buffer.getBuffer();
        æ.equal(isBuffer(buf), true);
        æ.equal(buffer.length, 0);
        æ.equal(buf.length, 0);
        return æ.done();
      });
      _ref1 = Array.prototype.slice(results);
      for (_i = 0, _len = _ref1.length; _i < _len; _i++) {
        result = _ref1[_i];
        buffer.write(result);
      }
      return buffer.end();
    },
    concat: function(æ) {
      var b1, b2, b3, concat;
      concat = BufferStream.fn.concat;
      b1 = new Buffer("a");
      b2 = new Buffer("bc");
      b3 = new Buffer("def");
      æ.equal(concat(b1, b2, b3).toString(), b1 + "" + b2 + "" + b3);
      æ.equal(concat(b2, b3).toString(), b2 + "" + b3);
      æ.equal(concat(b3).toString(), "" + b3);
      return æ.done();
    },
    split: function(æ) {
      var i, results, stream;
      stream = new BufferStream({
        encoding: 'utf8',
        size: 'flexible',
        split: ['//', ':']
      });
      stream.on('end', function() {
        æ.equal(stream.finished, true);
        æ.equal(i, 0);
        return æ.done();
      });
      results = [["buffer", ":"], ["stream", "//"], ["23:42", "//"]];
      i = 2;
      stream.on('split', function(chunk, token) {
        æ.deepEqual([chunk.toString(), token], results.shift());
        if (token === ':' || token === '//' && !(--i)) {
          return stream.disable(token);
        }
      });
      stream.on('data', function(chunk) {
        return æ.equal(chunk.toString(), "disabled");
      });
      æ.equal(stream.writable, true);
      æ.equal(stream.size, 'flexible');
      æ.equal(stream.write("buffer:stream//23:42//disabled"), true);
      return stream.end();
    },
    pipe: function(æ) {
      var buffer, filename, readme, stream;
      buffer = new BufferStream({
        size: 'flexible'
      });
      buffer.on('data', function(data) {
        throw 'up';
      });
      buffer.on('end', function() {
        æ.equal(buffer.enabled, true);
        æ.equal("" + (buffer.toString()) + "END", readme);
        return æ.done();
      });
      filename = path.join(__dirname, "..", "README.md");
      readme = "" + (readFileSync(filename)) + "END";
      stream = createReadStream(filename);
      return stream.pipe(buffer);
    },
    drainage: function(æ) {
      var buffer, results;
      buffer = new BufferStream({
        size: 'flexible',
        disabled: true
      });
      buffer.on('data', function(data) {
        return æ.equals(results.shift(), data.toString());
      });
      buffer.on('end', function() {
        æ.equals(0, results.length);
        return æ.done();
      });
      results = ["foo", "barbaz", "chaos"];
      buffer.write("foo");
      æ.equals(0, buffer.length);
      buffer.pause();
      buffer.write("bar");
      buffer.write("baz");
      buffer.resume();
      æ.equals(0, buffer.length);
      buffer.write("chaos");
      æ.equals(0, buffer.length);
      return buffer.end();
    }
  };

}).call(this);
