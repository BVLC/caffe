(function() {
  var BufferStream, Stream, fn, isArray, isBuffer, split, _ref,
    __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
    __hasProp = {}.hasOwnProperty,
    __extends = function(child, parent) { for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
    __slice = [].slice;

  Stream = require('stream').Stream;

  fn = require('./fn');

  _ref = [Array.isArray, Buffer.isBuffer], isArray = _ref[0], isBuffer = _ref[1];

  split = function() {
    var buflen, can_split, cur, found, i, pos, rest, splitter, _i, _len, _ref1, _ref2, _results;
    if (!this.buffer.length) {
      return;
    }
    can_split = this.enabled && this.splitters.length;
    _results = [];
    while (can_split) {
      cur = null;
      pos = buflen = this.buffer.length;
      _ref1 = this.splitters;
      for (_i = 0, _len = _ref1.length; _i < _len; _i++) {
        splitter = _ref1[_i];
        if (buflen < splitter.length) {
          continue;
        }
        i = fn.indexOf.call(this.buffer, splitter);
        if (i !== -1 && i < pos && i < buflen) {
          cur = splitter;
          pos = i;
        }
      }
      can_split = cur !== null;
      if (!can_split) {
        break;
      }
      _ref2 = fn.split(this.buffer, pos, cur.length), found = _ref2[0], rest = _ref2[1];
      this.buffer = rest;
      this.emit('split', found, cur);
      if (this.paused || !this.enabled || this.buffer.length === 0) {
        break;
      } else {
        _results.push(void 0);
      }
    }
    return _results;
  };

  BufferStream = (function(_super) {
    __extends(BufferStream, _super);

    function BufferStream(opts) {
      var _this = this;
      if (opts == null) {
        opts = {};
      }
      this.reset = __bind(this.reset, this);
      this.clear = __bind(this.clear, this);
      this.write = __bind(this.write, this);
      this.split = __bind(this.split, this);
      this.disable = __bind(this.disable, this);
      this.enable = __bind(this.enable, this);
      this.setSize = __bind(this.setSize, this);
      this.setEncoding = __bind(this.setEncoding, this);
      this.toString = __bind(this.toString, this);
      this.getBuffer = __bind(this.getBuffer, this);
      if (typeof opts === 'string') {
        opts = {
          encoding: opts
        };
      }
      if (opts.size == null) {
        opts.size = 'none';
      }
      if (opts.encoding == null) {
        opts.encoding = null;
      }
      if (opts.blocking == null) {
        opts.blocking = true;
      }
      this.size = opts.size;
      this.blocking = opts.blocking;
      this.splitters = [];
      this.__defineGetter__('length', function() {
        return _this.buffer.length;
      });
      this.setEncoding(opts.encoding);
      this.enabled = true;
      this.writable = true;
      this.readable = true;
      this.finished = false;
      this.paused = false;
      this.reset();
      BufferStream.__super__.constructor.call(this);
      if (opts.split != null) {
        if (isArray(opts.split)) {
          this.split(opts.split);
        } else {
          this.split(opts.split, function(data) {
            return this.emit('data', data);
          });
        }
      }
      if (opts.disabled) {
        this.disable();
      }
    }

    BufferStream.prototype.getBuffer = function() {
      return this.buffer;
    };

    BufferStream.prototype.toString = function() {
      var _ref1;
      return (_ref1 = this.buffer).toString.apply(_ref1, arguments);
    };

    BufferStream.prototype.setEncoding = function(encoding) {
      this.encoding = encoding;
    };

    BufferStream.prototype.setSize = function(size) {
      this.size = size;
      if (!this.paused && this.size === 'none') {
        return this.clear();
      }
    };

    BufferStream.prototype.enable = function() {
      return this.enabled = true;
    };

    BufferStream.prototype.disable = function() {
      var args, i, splitter, _i, _len;
      args = 1 <= arguments.length ? __slice.call(arguments, 0) : [];
      if (args.length === 1 && isArray(args[0])) {
        args = args[0];
      }
      for (_i = 0, _len = args.length; _i < _len; _i++) {
        splitter = args[_i];
        i = this.splitters.indexOf(splitter);
        if (i === -1) {
          continue;
        }
        this.splitters.splice(i, 1);
        if (!this.splitters.length) {
          break;
        }
      }
      if (!this.splitters.length) {
        this.enabled = false;
      }
      if (!args.length) {
        this.enabled = false;
        if (!this.paused) {
          return this.clear();
        }
      }
    };

    BufferStream.prototype.split = function() {
      var args, callback, splitter;
      args = 1 <= arguments.length ? __slice.call(arguments, 0) : [];
      if (args.length === 1 && isArray(args[0])) {
        args = args[0];
      }
      if (args.length === 2 && typeof args[1] === 'function') {
        splitter = args[0], callback = args[1];
        this.splitters.push(splitter);
        return this.on('split', function(_, token) {
          if (token === splitter) {
            return callback.apply(this, arguments);
          }
        });
      }
      return this.splitters = this.splitters.concat(args);
    };

    BufferStream.prototype.write = function(buffer, encoding) {
      var _this = this;
      if (!this.writable) {
        this.emit('error', new Error("Stream is not writable."));
        return false;
      }
      if (isBuffer(buffer)) {

      } else if (typeof buffer === 'string') {
        buffer = new Buffer(buffer, encoding != null ? encoding : this.encoding);
      } else {
        this.emit('error', new Error("Argument should be either a buffer or a string."));
      }
      if (this.buffer.length === 0) {
        this.buffer = buffer;
      } else {
        this.buffer = fn.concat(this.buffer, buffer);
      }
      if (this.paused) {
        return false;
      }
      if (this.size === 'none') {
        if (this.enabled) {
          split.call(this);
        }
        return this.clear();
      } else if (this.size === 'flexible') {
        if (this.enabled) {
          split.call(this);
        }
        if (this.finished) {
          return this.clear();
        }
        if (this.blocking) {
          if (!this.enabled) {
            this.clear();
          }
          return true;
        } else {
          process.nextTick(function() {
            return _this.emit('drain');
          });
          return false;
        }
      } else {
        throw new Error("not implemented yet :(");
      }
    };

    BufferStream.prototype.clear = function() {
      var buffer;
      if (!this.buffer.length) {
        return true;
      }
      buffer = this.buffer;
      this.reset();
      this.emit('data', buffer);
      return !this.paused;
    };

    BufferStream.prototype.reset = function() {
      if (typeof this.size === 'number') {
        return this.buffer = new Buffer(this.size);
      } else {
        return this.buffer = new Buffer(0);
      }
    };

    BufferStream.prototype.pause = function() {
      if (this.paused) {
        return;
      }
      this.paused = true;
      return this.emit('pause');
    };

    BufferStream.prototype.resume = function() {
      if (!this.paused) {
        return;
      }
      this.paused = false;
      split.call(this);
      if (this.paused) {
        return;
      }
      if (!this.enabled || this.size === 'none' || this.finished) {
        if (!this.clear()) {
          return;
        }
      }
      this.emit('drain');
      if (this.size === 'none') {
        this.emit('resume');
      }
      if (this.finished) {
        this.emit('end');
        return this.emit('close');
      }
    };

    BufferStream.prototype.end = function(data, encoding) {
      if (this.finished) {
        return;
      }
      this.finished = true;
      if (data != null) {
        this.write(data, encoding);
      }
      this.writable = false;
      if (!this.paused) {
        this.emit('end');
        return this.emit('close');
      }
    };

    return BufferStream;

  })(Stream);

  BufferStream.fn = fn;

  module.exports = BufferStream;

}).call(this);
