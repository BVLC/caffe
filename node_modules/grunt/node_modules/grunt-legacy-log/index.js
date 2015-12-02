/*
 * grunt
 * http://gruntjs.com/
 *
 * Copyright (c) 2014 "Cowboy" Ben Alman
 * Licensed under the MIT license.
 * https://github.com/gruntjs/grunt/blob/master/LICENSE-MIT
 */

'use strict';

// Nodejs libs.
var util = require('util');

// External libs.
var hooker = require('hooker');
// Requiring this here modifies the String prototype!
var colors = require('colors');
// The upcoming lodash 2.5+ should remove the need for underscore.string.
var _ = require('lodash');
_.str = require('underscore.string');
_.mixin(_.str.exports());
// TODO: ADD CHALK

var logUtils = require('grunt-legacy-log-utils');

function Log(options) {
  // This property always refers to the "base" logger.
  this.always = this;
  // Extend options.
  this.options = _.extend({}, {
    // Show colors in output?
    color: true,
    // Enable verbose-mode logging?
    verbose: false,
    // Enable debug logging statement?
    debug: false,
    // Where should messages be output?
    outStream: process.stdout,
    // NOTE: the color, verbose, debug options will be ignored if the
    // "grunt" option is specified! See the Log.prototype.option and
    // the Log.prototype.error methods for more info.
    grunt: null,
    // Where should output wrap? If null, use legacy Grunt defaults.
    maxCols: null,
    // Should logger start muted?
    muted: false,
  }, options);
  // True once anything has actually been logged.
  this.hasLogged = false;

  // Related verbose / notverbose loggers.
  this.verbose = new VerboseLog(this, true);
  this.notverbose = new VerboseLog(this, false);
  this.verbose.or = this.notverbose;
  this.notverbose.or = this.verbose;

  // Apparently, people have using grunt.log in interesting ways. Just bind
  // all methods so that "this" is irrelevant.
  if (this.options.grunt) {
    _.bindAll(this);
    _.bindAll(this.verbose);
    _.bindAll(this.notverbose);
  }
}
exports.Log = Log;

// Am I doing it wrong? :P
function VerboseLog(parentLog, verbose) {
  // Keep track of the original, base "Log" instance.
  this.always = parentLog;
  // This logger is either verbose (true) or notverbose (false).
  this._isVerbose = verbose;
}
util.inherits(VerboseLog, Log);

VerboseLog.prototype._write = function() {
  // Abort if not in correct verbose mode.
  if (Boolean(this.option('verbose')) !== this._isVerbose) { return; }
  // Otherwise... log!
  return VerboseLog.super_.prototype._write.apply(this, arguments);
};

// Create read/write accessors that prefer the parent log's properties (in
// the case of verbose/notverbose) to the current log's properties.
function makeSmartAccessor(name, isOption) {
  Object.defineProperty(Log.prototype, name, {
    enumerable: true,
    configurable: true,
    get: function() {
      return isOption ? this.always._options[name] : this.always['_' + name];
    },
    set: function(value) {
      if (isOption) {
        this.always._options[name] = value;
      } else {
        this.always['_' + name] = value;
      }
    },
  });
}
makeSmartAccessor('options');
makeSmartAccessor('hasLogged');
makeSmartAccessor('muted', true);

// Disable colors if --no-colors was passed.
Log.prototype.initColors = function() {
  if (this.option('no-color')) {
    // String color getters should just return the string.
    colors.mode = 'none';
    // Strip colors from strings passed to console.log.
    hooker.hook(console, 'log', function() {
      var args = _.toArray(arguments);
      return hooker.filter(this, args.map(function(arg) {
        return typeof arg === 'string' ? colors.stripColors(arg) : arg;
      }));
    });
  }
};

// Check for color, verbose, debug options through Grunt if specified,
// otherwise defer to options object properties.
Log.prototype.option = function(name) {
  if (this.options.grunt && this.options.grunt.option) {
    return this.options.grunt.option(name);
  }
  var no = name.match(/^no-(.+)$/);
  return no ? !this.options[no[1]] : this.options[name];
};

// Parse certain markup in strings to be logged.
Log.prototype._markup = function(str) {
  str = str || '';
  // Make _foo_ underline.
  str = str.replace(/(\s|^)_(\S|\S[\s\S]+?\S)_(?=[\s,.!?]|$)/g, '$1' + '$2'.underline);
  // Make *foo* bold.
  str = str.replace(/(\s|^)\*(\S|\S[\s\S]+?\S)\*(?=[\s,.!?]|$)/g, '$1' + '$2'.bold);
  return str;
};

// Similar to util.format in the standard library, however it'll always
// convert the first argument to a string and treat it as the format string.
Log.prototype._format = function(args) {
  args = _.toArray(args);
  if (args.length > 0) {
    args[0] = String(args[0]);
  }
  return util.format.apply(util, args);
};

Log.prototype._write = function(msg) {
  // Abort if muted.
  if (this.muted) { return; }
  // Actually write output.
  this.hasLogged = true;
  msg = msg || '';
  // Users should probably use the colors-provided methods, but if they
  // don't, this should strip extraneous color codes.
  if (this.option('no-color')) { msg = colors.stripColors(msg); }
  // Actually write to stdout.
  this.options.outStream.write(this._markup(msg));
};

Log.prototype._writeln = function(msg) {
  // Write blank line if no msg is passed in.
  this._write((msg || '') + '\n');
};

// Write output.
Log.prototype.write = function() {
  this._write(this._format(arguments));
  return this;
};

// Write a line of output.
Log.prototype.writeln = function() {
  this._writeln(this._format(arguments));
  return this;
};

Log.prototype.warn = function() {
  var msg = this._format(arguments);
  if (arguments.length > 0) {
    this._writeln('>> '.red + _.trim(msg).replace(/\n/g, '\n>> '.red));
  } else {
    this._writeln('ERROR'.red);
  }
  return this;
};
Log.prototype.error = function() {
  if (this.options.grunt && this.options.grunt.fail) {
    this.options.grunt.fail.errorcount++;
  }
  this.warn.apply(this, arguments);
  return this;
};
Log.prototype.ok = function() {
  var msg = this._format(arguments);
  if (arguments.length > 0) {
    this._writeln('>> '.green + _.trim(msg).replace(/\n/g, '\n>> '.green));
  } else {
    this._writeln('OK'.green);
  }
  return this;
};
Log.prototype.errorlns = function() {
  var msg = this._format(arguments);
  this.error(this.wraptext(this.options.maxCols || 77, msg));
  return this;
};
Log.prototype.oklns = function() {
  var msg = this._format(arguments);
  this.ok(this.wraptext(this.options.maxCols || 77, msg));
  return this;
};
Log.prototype.success = function() {
  var msg = this._format(arguments);
  this._writeln(msg.green);
  return this;
};
Log.prototype.fail = function() {
  var msg = this._format(arguments);
  this._writeln(msg.red);
  return this;
};
Log.prototype.header = function() {
  var msg = this._format(arguments);
  // Skip line before header, but not if header is the very first line output.
  if (this.hasLogged) { this._writeln(); }
  this._writeln(msg.underline);
  return this;
};
Log.prototype.subhead = function() {
  var msg = this._format(arguments);
  // Skip line before subhead, but not if subhead is the very first line output.
  if (this.hasLogged) { this._writeln(); }
  this._writeln(msg.bold);
  return this;
};
// For debugging.
Log.prototype.debug = function() {
  var msg = this._format(arguments);
  if (this.option('debug')) {
    this._writeln('[D] ' + msg.magenta);
  }
  return this;
};

// Write a line of a table.
Log.prototype.writetableln = function(widths, texts) {
  this._writeln(this.table(widths, texts));
  return this;
};

// Wrap a long line of text.
Log.prototype.writelns = function() {
  var msg = this._format(arguments);
  this._writeln(this.wraptext(this.options.maxCols || 80, msg));
  return this;
};

// Display flags in verbose mode.
Log.prototype.writeflags = function(obj, prefix) {
  var wordlist;
  if (Array.isArray(obj)) {
    wordlist = this.wordlist(obj);
  } else if (typeof obj === 'object' && obj) {
    wordlist = this.wordlist(Object.keys(obj).map(function(key) {
      var val = obj[key];
      return key + (val === true ? '' : '=' + JSON.stringify(val));
    }));
  }
  this._writeln((prefix || 'Flags') + ': ' + (wordlist || '(none)'.cyan));
  return this;
};

// Add static methods.
[
  'wordlist',
  'uncolor',
  'wraptext',
  'table',
].forEach(function(prop) {
  Log.prototype[prop] = exports[prop] = logUtils[prop];
});
