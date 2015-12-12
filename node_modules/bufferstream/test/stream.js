(function() {
  var BufferStream, createReadStream, isBuffer, path, readFileSync, _ref;

  path = require('path');

  _ref = require('fs'), createReadStream = _ref.createReadStream, readFileSync = _ref.readFileSync;

  BufferStream = require('../');

  isBuffer = Buffer.isBuffer;

  module.exports = {
    defaults: function(æ) {
      var buffer, result, results, _i, _len, _ref1;
      buffer = new BufferStream;
      æ.equal(buffer.finished, false);
      æ.equal(buffer.writable, true);
      æ.equal(buffer.readable, true);
      æ.equal(buffer.size, 'none');
      results = ["123", "bufferstream"];
      buffer.on('data', function(data) {
        return æ.equal(data, results.shift());
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
    '::enable': function(æ) {
      var buffer;
      buffer = new BufferStream({
        disabled: true
      });
      æ.equal("enabled=" + buffer.enabled, "enabled=false");
      buffer.enable();
      æ.equal("enabled=" + buffer.enabled, "enabled=true");
      return æ.done();
    },
    '::disable': function(æ) {
      var buffer;
      buffer = new BufferStream;
      æ.equal("enabled=" + buffer.enabled, "enabled=true");
      buffer.disable();
      æ.equal("enabled=" + buffer.enabled, "enabled=false");
      return æ.done();
    },
    shortcut: function(æ) {
      var results, stream;
      stream = new BufferStream({
        size: 'flexible',
        split: '\n'
      });
      stream.on('end', æ.done);
      results = ["a", "bc", "def"];
      stream.on('data', function(chunk) {
        return æ.equal(chunk.toString(), results.shift());
      });
      stream.write("a\nbc\ndef");
      return stream.end();
    },
    pipe: function(æ) {
      var buffer, filename, readme, stream;
      buffer = new BufferStream({
        size: 'flexible',
        split: '\n'
      });
      buffer.on('data', function(data) {
        return æ.equal(data.toString(), readme.shift());
      });
      buffer.on('end', function() {
        æ.equal(buffer.length, 0);
        æ.equal(buffer.enabled, true);
        æ.equal(buffer.toString(), "");
        æ.deepEqual(readme, ["END"]);
        return æ.done();
      });
      filename = path.join(__dirname, "..", "README.md");
      readme = ("" + (readFileSync(filename)) + "END").split('\n');
      stream = createReadStream(filename);
      return stream.pipe(buffer);
    },
    'pause/resume split': function(æ) {
      var results, stream;
      stream = new BufferStream({
        size: 'flexible'
      });
      stream.on('data', function(chunk) {
        return æ.equal(chunk.toString(), results.shift());
      });
      stream.on('end', function() {
        æ.equal(stream.length, 0);
        æ.equal(stream.toString(), "");
        return æ.done();
      });
      stream.split('/', function(part) {
        stream.pause();
        return process.nextTick(function() {
          results.push(part.toString());
          return stream.resume();
        });
      });
      results = "buffer".split('');
      stream.write("b/u/f/f/e/r/");
      return stream.end();
    }
  };

}).call(this);
