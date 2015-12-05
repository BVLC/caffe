var EMPTY_ARRAY = [];
var assert = require('assert');

function Key(entry, children, debug) {
  this.entry = entry;
  validateEntry(entry);
  this.children = children || EMPTY_ARRAY;
  this.debug = debug;
}

function validateEntry(entry) {
  assert(entry.fullPath, 'entry requires fullPath');
  assert(typeof entry.isDirectory === 'function', 'entry requires isDirectory function');
}

Key.prototype.toString = function() {
  return ' type: '     + (this.isDirectory() ? 'directory' : 'file') +
         ' fullPath: ' + this.entry.fullPath +
         ' path: '     + this.entry.relativePath+
         ' mode: '     + this.netry.mode +
         ' size: '     + this.netry.size +
         ' mtime: '    + this.netry.mtime;
};

function logNotEqual(current, next) {
  if (next) {
    current.debug(" cache eviction due to: \n     - {%o} \n     - {%o}", current, next);
  } else {
    current.debug(" cache empty, priming with: - {%o} ", next);
  }
}

Object.defineProperty(Key.prototype, 'fullPath', {
  get: function() {
    return this.entry.fullPath;
  }
});

Key.prototype.inspect = function() {
  return [
    this.entry.isDirectory() ? 'directory' : 'file',
    this.entry.fullPath,
    this.entry.relativePath,
    this.entry.mode,
    this.entry.size,
    this.entry.mtime
  ].join(', ');
};

Key.prototype.isDirectory = function() {
  return this.entry.isDirectory();
};

function isEntryEqual(a, b) {
  if (a.isDirectory() && b.isDirectory()) {
    return a.fullPath === b.fullPath;
  } else {
    return a.fullPath === b.fullPath &&
           a.mode     === b.mode &&
           a.size     === b.size &&
           a.mtime    === b.mtime;
  }
}

Key.prototype.isEqual = function(otherKey) {
  if (otherKey === undefined) {
    logNotEqual(this, otherKey);
    return false;
  }

  if (this.entry.isDirectory() && otherKey.entry.isDirectory()) {
    var children = this.children;
    var otherChildren = otherKey.children;

    if (children.length === otherChildren.length) {
      for (var i = 0; i < children.length; i++) {
        if (children[i].isEqual(otherChildren[i])) {
          // they are the same
        } else {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  // key represents a file, diff the file
  if (isEntryEqual(this.entry, otherKey.entry)) {
    return true;
  } else {
    logNotEqual(this, otherKey);
  }
};

module.exports = Key;
