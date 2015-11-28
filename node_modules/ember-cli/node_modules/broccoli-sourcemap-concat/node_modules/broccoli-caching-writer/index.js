'use strict';

var fs = require('fs');
var path = require('path');
var RSVP = require('rsvp');
var rimraf = RSVP.denodeify(require('rimraf'));
var helpers = require('broccoli-kitchen-sink-helpers');
var symlinkOrCopy = require('symlink-or-copy');
var assign = require('lodash-node/modern/object/assign');
var Plugin = require('broccoli-plugin');
var debugGenerator = require('debug');
var Key = require('./key');
var canUseInputFiles = require('./can-use-input-files');
var walkSync = require('walk-sync');

CachingWriter.prototype = Object.create(Plugin.prototype);
CachingWriter.prototype.constructor = CachingWriter;
function CachingWriter (inputNodes, options) {
  options = options || {};

  Plugin.call(this, inputNodes, {
    name: options.name,
    annotation: options.annotation,
    persistentOutput: true
  });

  this._cachingWriterPersistentOutput = !!options.persistentOutput;

  this._lastKeys = null;
  this._shouldBeIgnoredCache = Object.create(null);
  this._resetStats();

  this._cacheInclude = options.cacheInclude || [];
  this._cacheExclude = options.cacheExclude || [];
  this._inputFiles = options.inputFiles || {};

  if (!Array.isArray(this._cacheInclude)) {
    throw new Error('Invalid cacheInclude option, it must be an array or undefined.');
  }

  if (!Array.isArray(this._cacheExclude)) {
    throw new Error('Invalid cacheExclude option, it must be an array or undefined.');
  }
}

Object.defineProperty(CachingWriter.prototype, 'debug', {
  get: function() {
    return this._debug || (this._debug = debugGenerator(
      'broccoli-caching-writer:' +
        this._name +
        (this._annotation ? (' > [' + this._annotation + ']') : '')));
  }
});

CachingWriter.prototype._resetStats = function() {
  this._stats = {
    stats: 0,
    files: 0
  };
};

CachingWriter.prototype.getCallbackObject = function() {
  return {
    build: this._conditionalBuild.bind(this)
  };
};

CachingWriter.prototype._conditionalBuild = function () {
  var writer = this;
  var start = new Date();

  var invalidateCache = false;
  var dir;
  var lastKeys = [];

  if (!writer._lastKeys) {
    writer._lastKeys = [];
    // Force initial build even if inputNodes is []
    invalidateCache = true;
  }

  function shouldNotBeIgnored(relativePath) {
    /*jshint validthis:true */
    return !this.shouldBeIgnored(relativePath);
  }

  function keyForFile(relativePath) {
    var fullPath =  dir + '/' + relativePath;

    var stats = fs.statSync(fullPath);

    /*jshint validthis:true */
    return new Key({
      relativePath: relativePath,
      fullPath: dir + '/' + relativePath,
      basePath: dir,
      mode: stats.mode,
      size: stats.size,
      mtime: stats.mtime.getTime(),
      isDirectory: function() {
        return false;
      }
    }, undefined, this.debug);
  }
  for (var i = 0, l = writer.inputPaths.length; i < l; i++) {
    dir = writer.inputPaths[i];

    var files;

    if (canUseInputFiles(this._inputFiles)) {
      this.debug('using inputFiles directly');
      files = this._inputFiles.filter(shouldNotBeIgnored, this).map(keyForFile, this);
    } else {
      this.debug('walking %o', this.inputFiles);
      files = walkSync.entries(dir,  this.inputFiles).filter(entriesShouldNotBeIgnored, this).map(keyForEntry, this);
    }

    this._stats.files += files.length;

    var lastKey = writer._lastKeys[i];
    var key = keyForDir(dir, files, this.debug);

    lastKeys.push(key);

    if (!invalidateCache /* short circuit */ && !key.isEqual(lastKey)) {
      invalidateCache = true;
    }
  }

  this._stats.inputPaths = writer.inputPaths;
  this.debug('%o', this._stats);
  this.debug('derive cacheKey in %dms', new Date() - start);
  this._resetStats();

  if (invalidateCache) {
    start = new Date();
    writer._lastKeys = lastKeys;

    var promise = RSVP.Promise.resolve();
    if (!this._cachingWriterPersistentOutput) {
      promise = promise.then(function() {
        return rimraf(writer.outputPath);
      }).then(function() {
        fs.mkdirSync(writer.outputPath);
      }).finally(function() {

        this.debug('purge output in %dms', new Date() - start);
        start = new Date();
      }.bind(this));
    }
    return promise.then(function() {
      return writer.build();
    }).finally( function() {
      this.debug('rebuilding cache in %dms', new Date() - start);
    }.bind(this));
  }
};

// Takes in a path and { include, exclude }. Tests the path using regular expressions and
// returns true if the path does not match any exclude patterns AND matches atleast
// one include pattern.
CachingWriter.prototype.shouldBeIgnored = function (fullPath) {
  if (this._shouldBeIgnoredCache[fullPath] !== undefined) {
    return this._shouldBeIgnoredCache[fullPath];
  }

  var excludePatterns = this._cacheExclude;
  var includePatterns = this._cacheInclude;
  var i = null;

  // Check exclude patterns
  for (i = 0; i < excludePatterns.length; i++) {
    // An exclude pattern that returns true should be ignored
    if (excludePatterns[i].test(fullPath) === true) {
      return (this._shouldBeIgnoredCache[fullPath] = true);
    }
  }

  // Check include patterns
  if (includePatterns !== undefined && includePatterns.length > 0) {
    for (i = 0; i < includePatterns.length; i++) {
      // An include pattern that returns true (and wasn't excluded at all)
      // should _not_ be ignored
      if (includePatterns[i].test(fullPath) === true) {
        return (this._shouldBeIgnoredCache[fullPath] = false);
      }
    }

    // If no include patterns were matched, ignore this file.
    return (this._shouldBeIgnoredCache[fullPath] = true);
  }

  // Otherwise, don't ignore this file
  return (this._shouldBeIgnoredCache[fullPath] = false);
};

// Returns a list of matched files
CachingWriter.prototype.listFiles = function() {
  return this.listEntries().map(function(entry) {
    return entry.fullPath;
  });
};


// Returns a list of matched files
CachingWriter.prototype.listEntries = function() {
  function listEntries(keys, files) {
    for (var i=0; i< keys.length; i++) {
      var key = keys[i];
      if (key.isDirectory()) {
        var children = key.children;
        if(children && children.length > 0) {
          listEntries(children, files);
        }
      } else {
        files.push(key.entry);
      }
    }
    return files;
  }
  return listEntries(this._lastKeys, []);
};


module.exports = CachingWriter;

function keyForDir(dir, children, debug) {
  // no need to stat dir, we don't care about anything other then that they are
  // a dir
  return new Key({
    relativePath: '/',
    basePath: dir,
    fullPath: dir,
    mode: 16877,
    size: 0,
    mtime: 0,
    isDirectory: function() { return  true; }
  }, children, debug);
}

function entriesShouldNotBeIgnored(entry) {
  /*jshint validthis:true */
  return !this.shouldBeIgnored(entry.relativePath);
}

function keyForEntry(entry) {
  /*jshint validthis:true */
  return new Key(entry, undefined, this.debug);
}

