'use strict';

var fs = require('fs');
var path = require('path-posix');
var mkdirp = require('mkdirp');
var walkSync = require('walk-sync');
var Minimatch = require('minimatch').Minimatch;
var arrayEqual = require('array-equal');
var Plugin = require('broccoli-plugin');
var symlinkOrCopy = require('symlink-or-copy');
var debug = require('debug');
var FSTree = require('fs-tree-diff');
var rimraf = require('rimraf');
var BlankObject = require('blank-object');

function makeDictionary() {
  var cache = new BlankObject();

  cache['_dict'] = null;
  delete cache['_dict'];
  return cache;
}
// copied mostly from node-glob cc @isaacs
function isNotAPattern(pattern) {
  var set = new Minimatch(pattern).set;
  if (set.length > 1) {
    return false;
  }

  for (var j = 0; j < set[0].length; j++) {
    if (typeof set[0][j] !== 'string') {
      return false;
    }
  }

  return true;
}

Funnel.prototype = Object.create(Plugin.prototype);
Funnel.prototype.constructor = Funnel;
function Funnel(inputNode, options) {
  if (!(this instanceof Funnel)) { return new Funnel(inputNode, options); }

  Plugin.call(this, [inputNode], options);

  this._persistentOutput = true;

  this._includeFileCache = makeDictionary();
  this._destinationPathCache = makeDictionary();
  this._currentTree = new FSTree();

  var keys = Object.keys(options || {});
  for (var i = 0, l = keys.length; i < l; i++) {
    var key = keys[i];
    this[key] = options[key];
  }

  this.destDir = this.destDir || '/';
  this.count = 0;

  if (this.files && typeof this.files === 'function') {
    // Save dynamic files func as a different variable and let the rest of the code
    // still assume that this.files is always an array.
    this._dynamicFilesFunc = this.files;
    delete this.files;
  } else if (this.files && !Array.isArray(this.files)) {
    throw new Error('Invalid files option, it must be an array or function (that returns an array).');
  }

  if ((this.files || this._dynamicFilesFunc) && (this.include || this.exclude)) {
    throw new Error('Cannot pass files option (array or function) and a include/exlude filter. You can only have one or the other');
  }

  if (this.files) {
    if (this.files.filter(isNotAPattern).length !== this.files.length) {
      console.warn('broccoli-funnel does not support `files:` option with globs, please use `include:` instead');
      this.include = this.files;
      this.files = undefined;
    }
  }

  this._setupFilter('include');
  this._setupFilter('exclude');

  this._matchedWalk = this.include && this.include.filter(function(a) {
    return a instanceof Minimatch;
  }).length === this.include.length;

  this._instantiatedStack = (new Error()).stack;
  this._buildStart = undefined;
}

Funnel.prototype._debugName = function() {
  return this.description || this._annotation || this.name || this.constructor.name;
};

Funnel.prototype._debug = function(message) {
  debug('broccoli-funnel:' + (this._debugName())).apply(null, arguments);
};

Funnel.prototype._setupFilter = function(type) {
  if (!this[type]) {
    return;
  }

  if (!Array.isArray(this[type])) {
    throw new Error('Invalid ' + type + ' option, it must be an array. You specified `' + typeof this[type] + '`.');
  }

  // Clone the filter array so we are not mutating an external variable
  var filters = this[type] = this[type].slice(0);

  for (var i = 0, l = filters.length; i < l; i++) {
    filters[i] = this._processPattern(filters[i]);
  }
};

Funnel.prototype._processPattern = function(pattern) {
  if (pattern instanceof RegExp) {
    return pattern;
  }

  var type = typeof pattern;

  if (type === 'string') {
    return new Minimatch(pattern);
  }

  if (type === 'function') {
    return pattern;
  }

  throw new Error('include/exclude patterns can be a RegExp, glob string, or function. You supplied `' + typeof pattern +'`.');
};

Funnel.prototype.shouldLinkRoots = function() {
  return !this.files && !this.include && !this.exclude && !this.getDestinationPath;
};

Funnel.prototype.build = function() {
  this._buildStart = new Date();
  this.destPath = path.join(this.outputPath, this.destDir);
  if (this.destPath[this.destPath.length -1] === '/') {
    this.destPath = this.destPath.slice(0, -1);
  }

  var inputPath = this.inputPaths[0];
  if (this.srcDir) {
    inputPath = path.join(inputPath, this.srcDir);
  }

  if (this._dynamicFilesFunc) {
    this.lastFiles = this.files;
    this.files = this._dynamicFilesFunc() || [];

    // Blow away the include cache if the list of files is new
    if (this.lastFiles !== undefined && !arrayEqual(this.lastFiles, this.files)) {
      this._includeFileCache = makeDictionary();
    }
  }

  var linkedRoots = false;
  if (this.shouldLinkRoots()) {
    linkedRoots = true;
    if (fs.existsSync(inputPath)) {
      rimraf.sync(this.outputPath);
      this._copy(inputPath, this.destPath);
    } else if (this.allowEmpty) {
      mkdirp.sync(this.destPath);
    }
  } else {
    this.processFilters(inputPath);
  }

  this._debug('build, %o', {
    in: new Date() - this._buildStart + 'ms',
    linkedRoots: linkedRoots,
    inputPath: inputPath,
    destPath: this.destPath
  });
};

function ensureRelative(string) {
  if (string.charAt(0) === '/') {
    return string.substring(1);
  }
  return string;
}

Funnel.prototype._processEntries = function(entries) {
  return entries.filter(function(entry) {
    // support the second set of filters walk-sync does not support
    //   * regexp
    //   * excludes
    return this.includeFile(entry.relativePath);
  }, this).map(function(entry) {

    var relativePath = entry.relativePath;

    entry.relativePath = this.lookupDestinationPath(relativePath);

    this.outputToInputMappings[entry.relativePath] = relativePath;

    return entry;
  }, this);
};

Funnel.prototype._processPaths  = function(paths, outputToInputMappings) {
  return paths.
    slice(0).
    filter(this.includeFile, this).
    map(function(relativePath) {
      var output = this.lookupDestinationPath(relativePath);
      this.outputToInputMappings[output] = relativePath;
      return output;
    }, this);
};

Funnel.prototype.processFilters = function(inputPath) {
  var nextTree;

  this.outputToInputMappings = {}; // we allow users to rename files

  if (this.files && !this.exclude && !this.include) {
    // clone to be compatible with walkSync
    nextTree = FSTree.fromPaths(this._processPaths(this.files));
  } else {
    var entries;

    if (this._matchedWalk) {
      entries = walkSync.entries(inputPath, this.include);
    } else {
      entries = walkSync.entries(inputPath);
    }

    nextTree = new FSTree({
      entries: this._processEntries(entries)
    });
  }

  var patch = this._currentTree.calculatePatch(nextTree);

  this._currentTree = nextTree;

  this._debug('patch size: %d', patch.length);

  var outputPath = this.outputPath;

  patch.forEach(function(entry) {
    this._applyPatch(entry, inputPath, outputPath);
  }, this);

  var count = nextTree.size;

  this._debug('processFilters %o', {
    in: new Date() - this._buildStart + 'ms',
    filesFound: this._currentTree.size,
    filesProcessed: count,
    operations: patch.length,
    inputPath: inputPath,
    destPath: this.destPath
  });
};

Funnel.prototype._applyPatch = function applyPatch(entry, inputPath, _outputPath) {
  var outputToInput = this.outputToInputMappings;
  var operation = entry[0];
  var outputRelative = entry[1];

  if (!outputRelative) {
    // broccoli itself maintains the roots, we can skip any operation on them
    return;
  }

  var outputPath = _outputPath + '/' + outputRelative;

  this._debug('%s %s', operation, outputPath);

  if (operation === 'change') {
    operation = 'create';
  }

  switch (operation) {
    case 'unlink' :
      fs.unlinkSync(outputPath);
    break;
    case 'rmdir'  :
      fs.rmdirSync(outputPath);
    break;
    case 'mkdir'  :
      fs.mkdirSync(outputPath);
    break;
    case 'create'/* also change */ :
      var relativePath = outputToInput[outputRelative];
      if (relativePath === undefined) {
        relativePath = outputToInput['/' + outputRelative];
      }
      this.processFile(inputPath + '/' + relativePath, outputPath, relativePath);
      break;
    default: throw new Error('Unknown operation: ' + operation);
  }
};

Funnel.prototype.lookupDestinationPath = function(relativePath) {
  if (this._destinationPathCache[relativePath] !== undefined) {
    return this._destinationPathCache[relativePath];
  }

  // the destDir is absolute to prevent '..' above the output dir
  if (this.getDestinationPath) {
    return this._destinationPathCache[relativePath] = ensureRelative(path.join(this.destDir, this.getDestinationPath(relativePath)));
  }

  return this._destinationPathCache[relativePath] = ensureRelative(path.join(this.destDir, relativePath));
};

Funnel.prototype.includeFile = function(relativePath) {
  var includeFileCache = this._includeFileCache;

  if (includeFileCache[relativePath] !== undefined) {
    return includeFileCache[relativePath];
  }

  // do not include directories, only files
  if (relativePath[relativePath.length - 1] === '/') {
    return includeFileCache[relativePath] = false;
  }

  var i, l, pattern;

  // Check for specific files listing
  if (this.files) {
    return includeFileCache[relativePath] = this.files.indexOf(relativePath) > -1;
  }

  // Check exclude patterns
  if (this.exclude) {
    for (i = 0, l = this.exclude.length; i < l; i++) {
      // An exclude pattern that returns true should be ignored
      pattern = this.exclude[i];

      if (this._matchesPattern(pattern, relativePath)) {
        return includeFileCache[relativePath] = false;
      }
    }
  }

  // Check include patterns
  if (this.include && this.include.length > 0) {
    for (i = 0, l = this.include.length; i < l; i++) {
      // An include pattern that returns true (and wasn't excluded at all)
      // should _not_ be ignored
      pattern = this.include[i];

      if (this._matchesPattern(pattern, relativePath)) {
        return includeFileCache[relativePath] = true;
      }
    }

    // If no include patterns were matched, ignore this file.
    return includeFileCache[relativePath] = false;
  }

  // Otherwise, don't ignore this file
  return includeFileCache[relativePath] = true;
};

Funnel.prototype._matchesPattern = function(pattern, relativePath) {
  if (pattern instanceof RegExp) {
    return pattern.test(relativePath);
  } else if (pattern instanceof Minimatch) {
    return pattern.match(relativePath);
  } else if (typeof pattern === 'function') {
    return pattern(relativePath);
  }

  throw new Error('Pattern `' + pattern + '` was not a RegExp, Glob, or Function.');
};

Funnel.prototype.processFile = function(sourcePath, destPath /*, relativePath */) {
  this._copy(sourcePath, destPath);
};

Funnel.prototype._copy = function(sourcePath, destPath) {
  var destDir = path.dirname(destPath);

  try {
    symlinkOrCopy.sync(sourcePath, destPath);
  } catch(e) {
    if (!fs.existsSync(destDir)) {
      mkdirp.sync(destDir);
    }
    try {
      fs.unlinkSync(destPath);
    } catch(e) {

    }
    symlinkOrCopy.sync(sourcePath, destPath);
  }
};

module.exports = Funnel;
