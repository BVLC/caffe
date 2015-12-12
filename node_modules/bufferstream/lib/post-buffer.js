(function() {
  var BufferStream, PostBuffer,
    __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };

  BufferStream = require('./buffer-stream');

  PostBuffer = (function() {
    function PostBuffer(req, opts) {
      var _this = this;
      if (opts == null) {
        opts = {};
      }
      this.pipe = __bind(this.pipe, this);
      this.onEnd = __bind(this.onEnd, this);
      this.callback = null;
      this.got_all_data = false;
      if (opts.size == null) {
        opts.size = 'flexible';
      }
      this.stream = new BufferStream(opts);
      req.on('end', function() {
        _this.got_all_data = true;
        return typeof _this.callback === "function" ? _this.callback(_this.stream.buffer) : void 0;
      });
      req.pipe(this.stream);
    }

    PostBuffer.prototype.onEnd = function(callback) {
      this.callback = callback;
      if (this.got_all_data) {
        return this.callback(this.stream.buffer);
      }
    };

    PostBuffer.prototype.pipe = function(dest, options) {
      this.stream.pipe(dest, options);
      this.stream.setSize('none');
      return dest;
    };

    return PostBuffer;

  })();

  module.exports = PostBuffer;

}).call(this);
