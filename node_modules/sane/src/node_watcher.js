'use strict';

var fs = require('fs');
var path = require('path');
var walker = require('walker');
var common = require('./common');
var platform = require('os').platform();
var EventEmitter = require('events').EventEmitter;

/**
 * Constants
 */

var DEFAULT_DELAY = common.DEFAULT_DELAY;
var CHANGE_EVENT = common.CHANGE_EVENT;
var DELETE_EVENT = common.DELETE_EVENT;
var ADD_EVENT = common.ADD_EVENT;
var ALL_EVENT = common.ALL_EVENT;

/**
 * Export `NodeWatcher` class.
 */

module.exports = NodeWatcher;

/**
 * Watches `dir`.
 *
 * @class NodeWatcher
 * @param String dir
 * @param {Object} opts
 * @public
 */

function NodeWatcher(dir, opts) {
  opts = common.assignOptions(this, opts);

  this.watched = Object.create(null);
  this.changeTimers = Object.create(null);
  this.dirRegistery = Object.create(null);
  this.root = path.resolve(dir);
  this.watchdir = this.watchdir.bind(this);
  this.register = this.register.bind(this);

  this.watchdir(this.root);
  recReaddir(
    this.root,
    this.watchdir,
    this.register,
    this.emit.bind(this, 'ready')
  );
}

NodeWatcher.prototype.__proto__ = EventEmitter.prototype;

/**
 * Register files that matches our globs to know what to type of event to
 * emit in the future.
 *
 * Registery looks like the following:
 *
 *  dirRegister => Map {
 *    dirpath => Map {
 *       filename => true
 *    }
 *  }
 *
 * @param {string} filepath
 * @return {boolean} whether or not we have registered the file.
 * @private
 */

NodeWatcher.prototype.register = function(filepath) {
  var relativePath = path.relative(this.root, filepath);
  if (!common.isFileIncluded(this.globs, this.dot, relativePath)) {
    return false;
  }

  var dir = path.dirname(filepath);
  if (!this.dirRegistery[dir]) {
    this.dirRegistery[dir] = Object.create(null);
  }

  var filename = path.basename(filepath);
  this.dirRegistery[dir][filename] = true;

  return true;
};

/**
 * Removes a file from the registery.
 *
 * @param {string} filepath
 * @private
 */

NodeWatcher.prototype.unregister = function(filepath) {
  var dir = path.dirname(filepath);
  if (this.dirRegistery[dir]) {
    var filename = path.basename(filepath);
    delete this.dirRegistery[dir][filename];
  }
};

/**
 * Removes a dir from the registery.
 *
 * @param {string} dirpath
 * @private
 */

NodeWatcher.prototype.unregisterDir = function(dirpath) {
  if (this.dirRegistery[dirpath]) {
    delete this.dirRegistery[dirpath];
  }
};

/**
 * Checks if a file or directory exists in the registery.
 *
 * @param {string} fullpath
 * @return {boolean}
 * @private
 */

NodeWatcher.prototype.registered = function(fullpath) {
  var dir = path.dirname(fullpath);
  return this.dirRegistery[fullpath] ||
    this.dirRegistery[dir] && this.dirRegistery[dir][path.basename(fullpath)];
};

/**
 * Watch a directory.
 *
 * @param {string} dir
 * @private
 */

NodeWatcher.prototype.watchdir = function(dir) {
  if (this.watched[dir]) {
    return;
  }

  var watcher = fs.watch(
    dir,
    { persistent: true },
    this.normalizeChange.bind(this, dir)
  );
  this.watched[dir] = watcher;

  // Workaround Windows node issue #4337.
  if (platform === 'win32') {
    watcher.on('error', function(error) {
      if (error.code !== 'EPERM') {
        throw error;
      }
    });
  }

  if (this.root !== dir) {
    this.register(dir);
  }
};

/**
 * Stop watching a directory.
 *
 * @param {string} dir
 * @private
 */

NodeWatcher.prototype.stopWatching = function(dir) {
  if (this.watched[dir]) {
    this.watched[dir].close();
    delete this.watched[dir];
  }
};

/**
 * End watching.
 *
 * @public
 */

NodeWatcher.prototype.close = function(callback) {
  Object.keys(this.watched).forEach(this.stopWatching, this);
  this.removeAllListeners();
  if (typeof callback === 'function') {
    setImmediate(callback.bind(null, null, true));
  }
};

/**
 * On some platforms, as pointed out on the fs docs (most likely just win32)
 * the file argument might be missing from the fs event. Try to detect what
 * change by detecting if something was deleted or the most recent file change.
 *
 * @param {string} dir
 * @param {string} event
 * @param {string} file
 * @public
 */

NodeWatcher.prototype.detectChangedFile = function(dir, event, callback) {
  if (!this.dirRegistery[dir]) {
    return;
  }

  var found = false;
  var closest = {mtime: 0};
  var c = 0;
  Object.keys(this.dirRegistery[dir]).forEach(function(file, i, arr) {
    fs.lstat(path.join(dir, file), function(error, stat) {
      if (found) {
        return;
      }

      if (error) {
        if (error.code === 'ENOENT' ||
            (platform === 'win32' && error.code === 'EPERM')) {
          found = true;
          callback(file);
        } else {
          this.emit('error', error);
        }
      } else {
        if (stat.mtime > closest.mtime) {
          stat.file = file;
          closest = stat;
        }
        if (arr.length === ++c) {
          callback(closest.file);
        }
      }
    }.bind(this));
  }, this);
};

/**
 * Normalize fs events and pass it on to be processed.
 *
 * @param {string} dir
 * @param {string} event
 * @param {string} file
 * @public
 */

NodeWatcher.prototype.normalizeChange = function(dir, event, file) {
  if (!file) {
    this.detectChangedFile(dir, event, function(actualFile) {
      if (actualFile) {
        this.processChange(dir, event, actualFile);
      }
    }.bind(this));
  } else {
    this.processChange(dir, event, path.normalize(file));
  }
};

/**
 * Process changes.
 *
 * @param {string} dir
 * @param {string} event
 * @param {string} file
 * @public
 */

NodeWatcher.prototype.processChange = function(dir, event, file) {
  var fullPath = path.join(dir, file);
  var relativePath = path.join(path.relative(this.root, dir), file);
  fs.lstat(fullPath, function(error, stat) {
    if (error && error.code !== 'ENOENT') {
      this.emit('error', error);
    } else if (!error && stat.isDirectory()) {
      // win32 emits usless change events on dirs.
      if (event !== 'change') {
        this.watchdir(fullPath);
        this.emitEvent(ADD_EVENT, relativePath, stat);
      }
    } else {
      var registered = this.registered(fullPath);
      if (error && error.code === 'ENOENT') {
        this.unregister(fullPath);
        this.stopWatching(fullPath);
        this.unregisterDir(fullPath);
        if (registered) {
          this.emitEvent(DELETE_EVENT, relativePath);
        }
      } else if (registered) {
        this.emitEvent(CHANGE_EVENT, relativePath, stat);
      } else {
        if (this.register(fullPath)) {
          this.emitEvent(ADD_EVENT, relativePath, stat);
        }
      }
    }
  }.bind(this));
};

/**
 * Triggers a 'change' event after debounding it to take care of duplicate
 * events on os x.
 *
 * @private
 */

NodeWatcher.prototype.emitEvent = function(type, file, stat) {
  var key = type + '-' + file;
  clearTimeout(this.changeTimers[key]);
  this.changeTimers[key] = setTimeout(function() {
    delete this.changeTimers[key];
    this.emit(type, file, this.root, stat);
    this.emit(ALL_EVENT, type, file, this.root, stat);
  }.bind(this), DEFAULT_DELAY);
};

/**
 * Traverse a directory recursively calling `callback` on every directory.
 *
 * @param {string} dir
 * @param {function} callback
 * @param {function} endCallback
 * @private
 */

function recReaddir(dir, dirCallback, fileCallback, endCallback) {
  walker(dir)
    .on('dir', normalizeProxy(dirCallback))
    .on('file', normalizeProxy(fileCallback))
    .on('end', function() {
      if (platform === 'win32') {
        setTimeout(endCallback, 1000);
      } else {
        endCallback();
      }
    });
}

/**
 * Returns a callback that when called will normalize a path and call the
 * original callback
 *
 * @param {function} callback
 * @return {function}
 * @private
 */

function normalizeProxy(callback) {
  return function(filepath) {
    return callback(path.normalize(filepath));
  };
}
