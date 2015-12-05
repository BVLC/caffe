'use strict';

var fs = require('fs');
var path = require('path');
var assert = require('assert');
var common = require('./common');
var watchman = require('fb-watchman');
var EventEmitter = require('events').EventEmitter;

/**
 * Constants
 */

var CHANGE_EVENT = common.CHANGE_EVENT;
var DELETE_EVENT = common.DELETE_EVENT;
var ADD_EVENT = common.ADD_EVENT;
var ALL_EVENT = common.ALL_EVENT;
var SUB_NAME = 'sane-sub';

/**
 * Export `WatchmanWatcher` class.
 */

module.exports = WatchmanWatcher;


/**
 * Watches `dir`.
 *
 * @class PollWatcher
 * @param String dir
 * @param {Object} opts
 * @public
 */

function WatchmanWatcher(dir, opts) {
  opts = common.assignOptions(this, opts);
  this.root = path.resolve(dir);
  this.init();
}

WatchmanWatcher.prototype.__proto__ = EventEmitter.prototype;

/**
 * Run the watchman `watch` command on the root and subscribe to changes.
 *
 * @private
 */

WatchmanWatcher.prototype.init = function() {
  if (this.client) {
    this.client.removeAllListeners();
  }

  var self = this;
  this.client = new watchman.Client();
  this.client.on('error', function(error) {
    self.emit('error', error);
  });
  this.client.on('subscription', this.handleChangeEvent.bind(this));
  this.client.on('end', function() {
    console.warn('[sane] Warning: Lost connection to watchman, reconnecting..');
    self.init();
  });

  this.watchProjectInfo = null;

  function getWatchRoot() {
    return self.watchProjectInfo ? self.watchProjectInfo.root : self.root;
  }

  function onCapability(error, resp) {
    if (handleError(self, error)) {
      // The Watchman watcher is unusable on this system, we cannot continue
      return;
    }

    handleWarning(resp);

    self.capabilities = resp.capabilities;

    if (self.capabilities.relative_root) {
      self.client.command(
        ['watch-project', getWatchRoot()], onWatchProject
      );
    } else {
      self.client.command(['watch', getWatchRoot()], onWatch);
    }
  }

  function onWatchProject(error, resp) {
    if (handleError(self, error)) {
      return;
    }

    handleWarning(resp);

    self.watchProjectInfo = {
      root: resp.watch,
      relativePath: resp.relative_path ? resp.relative_path : ''
    };

    self.client.command(['clock', getWatchRoot()], onClock);
  }


  function onWatch(error, resp) {
    if (handleError(self, error)) {
      return;
    }

    handleWarning(resp);

    self.client.command(['clock', getWatchRoot()], onClock);
  }

  function onClock(error, resp) {
    if (handleError(self, error)) {
      return;
    }

    handleWarning(resp);

    var options = {
      fields: ['name', 'exists', 'new'],
      since: resp.clock
    };

    // If the server has the wildmatch capability available it supports
    // the recursive **/*.foo style match and we can offload our globs
    // to the watchman server.  This saves both on data size to be
    // communicated back to us and compute for evaluating the globs
    // in our node process.
    if (self.capabilities.wildmatch) {
      if (self.globs.length === 0) {
        if (!self.dot) {
          // Make sure we honor the dot option if even we're not using globs.
          options.expression = ['match', '*', 'basename', {
            includedotfiles: false
          }];
        }
      } else {
        options.expression = ['anyof'];
        for (var i in self.globs) {
          options.expression.push(['match', self.globs[i], 'wholename', {
            includedotfiles: self.dot
          }]);
        }
      }
    }

    if (self.capabilities.relative_root) {
      options.relative_root = self.watchProjectInfo.relativePath;
    }

    self.client.command(
      ['subscribe', getWatchRoot(), SUB_NAME, options],
      onSubscribe
    );
  }

  function onSubscribe(error, resp) {
    if (handleError(self, error)) {
      return;
    }

    handleWarning(resp);

    self.emit('ready');
  }

  self.client.capabilityCheck({
      optional:['wildmatch', 'relative_root']
    },
    onCapability);
};

/**
 * Handles a change event coming from the subscription.
 *
 * @param {Object} resp
 * @private
 */

WatchmanWatcher.prototype.handleChangeEvent = function(resp) {
  assert.equal(resp.subscription, SUB_NAME, 'Invalid subscription event.');
  resp.files.forEach(this.handleFileChange, this);
};

/**
 * Handles a single change event record.
 *
 * @param {Object} changeDescriptor
 * @private
 */

WatchmanWatcher.prototype.handleFileChange = function(changeDescriptor) {
  var self = this;
  var absPath;
  var relativePath;

  if (this.capabilities.relative_root) {
    relativePath = changeDescriptor.name;
    absPath = path.join(
      this.watchProjectInfo.root,
      this.watchProjectInfo.relativePath,
      relativePath
    );
  } else {
    absPath = path.join(this.root, changeDescriptor.name);
    relativePath = changeDescriptor.name;
  }

  if (!self.capabilities.wildmatch &&
      !common.isFileIncluded(this.globs, this.dot, relativePath)) {
    return;
  }

  if (!changeDescriptor.exists) {
    self.emitEvent(DELETE_EVENT, relativePath, self.root);
  } else {
    fs.lstat(absPath, function(error, stat) {
      // Files can be deleted between the event and the lstat call
      // the most reliable thing to do here is to ignore the event.
      if (error && error.code === 'ENOENT') {
        return;
      }

      if (handleError(self, error)) {
        return;
      }

      var eventType = changeDescriptor.new ? ADD_EVENT : CHANGE_EVENT;

      // Change event on dirs are mostly useless.
      if (!(eventType === CHANGE_EVENT && stat.isDirectory())) {
        self.emitEvent(eventType, relativePath, self.root, stat);
      }
    });
  }
};

/**
 * Dispatches the event.
 *
 * @param {string} eventType
 * @param {string} filepath
 * @param {string} root
 * @param {fs.Stat} stat
 * @private
 */

WatchmanWatcher.prototype.emitEvent = function(
  eventType,
  filepath,
  root,
  stat
) {
  this.emit(eventType, filepath, root, stat);
  this.emit(ALL_EVENT, eventType, filepath, root, stat);
};

/**
 * Closes the watcher.
 *
 * @param {function} callback
 * @private
 */

WatchmanWatcher.prototype.close = function(callback) {
  this.client.removeAllListeners();
  this.client.end();
  callback && callback(null, true);
};

/**
 * Handles an error and returns true if exists.
 *
 * @param {WatchmanWatcher} self
 * @param {Error} error
 * @private
 */

function handleError(self, error) {
  if (error != null) {
    self.emit('error', error);
    return true;
  } else {
    return false;
  }
}

/**
 * Handles a warning in the watchman resp object.
 *
 * @param {object} resp
 * @private
 */

function handleWarning(resp) {
  if ('warning' in resp) {
    console.warn(resp.warning);
    return true;
  } else {
    return false;
  }
}
