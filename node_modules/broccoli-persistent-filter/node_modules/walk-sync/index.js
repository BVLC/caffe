'use strict';

var fs = require('fs');
var MatcherCollection = require('matcher-collection');
var path = require('path');

function handleOptions(_options) {
  var options = {};
  if (Array.isArray(_options)) {
    options.globs = _options;
  } else if (_options) {
    options = _options;
  }

  return options;
}

function handleRelativePath(_relativePath) {
  if (_relativePath == null) {
    return '';
  } else if (_relativePath.slice(-1) !== '/') {
    return _relativePath + '/';
  }
}

function ensurePosix(filepath) {
  if (path.sep !== '/') {
    return filepath.split(path.sep).join('/');
  }

  return filepath;
}

module.exports = walkSync;
function walkSync(baseDir, _options) {
  var options = handleOptions(_options);

  return _walkSync(baseDir, options).map(function(entry) {
    return entry.relativePath;
  });
}

module.exports.entries = function entries(baseDir, _options) {
  var options = handleOptions(_options);


  return _walkSync(ensurePosix(baseDir), options);
};

function _walkSync(baseDir, options, _relativePath) {
  // Inside this function, prefer string concatenation to the slower path.join
  // https://github.com/joyent/node/pull/6929
  var relativePath = handleRelativePath(_relativePath);
  var globs = options.globs;
  var m;

  if (globs) {
    m = new MatcherCollection(globs);
  }

  var results = [];
  if (m && !m.mayContain(relativePath)) {
    return results;
  }

  var entries = fs.readdirSync(baseDir + '/' + relativePath).sort();
  for (var i = 0; i < entries.length; i++) {
    var entryRelativePath = relativePath + entries[i];
    var fullPath = baseDir + '/' + entryRelativePath;
    var stats = getStat(fullPath);

    if (stats && stats.isDirectory()) {
      if (options.directories !== false && (!m || m.match(entryRelativePath))) {
        results.push(new Entry(entryRelativePath + '/', baseDir, stats.mode, stats.size, stats.mtime.getTime()));
      }
      results = results.concat(_walkSync(baseDir, options, entryRelativePath));
    } else {
      if (!m || m.match(entryRelativePath)) {
        results.push(new Entry(entryRelativePath, baseDir, stats && stats.mode, stats && stats.size, stats && stats.mtime.getTime()));
      }
    }
  }
  return results;
}

function Entry(relativePath, basePath, mode, size, mtime) {
  this.relativePath = relativePath;
  this.basePath = basePath;
  this.mode = mode;
  this.size = size;
  this.mtime = mtime;
}

Object.defineProperty(Entry.prototype, 'fullPath', {
  get: function() {
    return this.basePath + '/' + this.relativePath;
  }
});

Entry.prototype.isDirectory = function () {
  return (this.mode & 61440) === 16384;
};

function getStat(path) {
  var stat;

  try {
    stat = fs.statSync(path);
  } catch(error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }
  }

  return stat;
}
