'use strict';

var fs = require('fs');
var path = require('path');
var watch = require('watch');
var common = require('../src/common');
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
 * Export `PluginTestWatcher` class.
 *
 * This class is based on the PollWatcher but emits a special event to signal that it
 * is the watcher that is used to verify that the plugin system is working
 *
 */

module.exports = PluginTestWatcher;

/**
 * Watches `dir`.
 *
 * @class PluginTestWatcher
 * @param String dir
 * @param {Object} opts
 * @public
 */

function PluginTestWatcher(dir, opts) {
  opts = common.assignOptions(this, opts);

  this.watched = Object.create(null);
  this.root = path.resolve(dir);

  watch.createMonitor(
    this.root,
    { interval: opts.interval || DEFAULT_DELAY,
      filter: this.filter.bind(this)
    },
    this.init.bind(this)
  );
}

PluginTestWatcher.prototype.__proto__ = EventEmitter.prototype;

/**
 * Given a fullpath of a file or directory check if we need to watch it.
 *
 * @param {string} filepath
 * @param {object} stat
 * @private
 */

PluginTestWatcher.prototype.filter = function(filepath, stat) {
  return stat.isDirectory() || common.isFileIncluded(
    this.globs,
    this.dot,
    path.relative(this.root, filepath)
  );
};

/**
 * Initiate the polling file watcher with the event emitter passed from
 * `watch.watchTree`.
 *
 * @param {EventEmitter} monitor
 * @public
 */

PluginTestWatcher.prototype.init = function(monitor) {
  this.watched = monitor.files;
  monitor.on('changed', this.emitEvent.bind(this, CHANGE_EVENT));
  monitor.on('removed', this.emitEvent.bind(this, DELETE_EVENT));
  monitor.on('created', this.emitEvent.bind(this, ADD_EVENT));
  // 1 second wait because mtime is second-based.
  setTimeout(this.emit.bind(this, 'ready'), 1000);

  // This event is fired to note that this watcher is the one from the plugin system
  setTimeout(this.emit.bind(this, 'is-test-plugin'), 1);
};

/**
 * Transform and emit an event comming from the poller.
 *
 * @param {EventEmitter} monitor
 * @public
 */

PluginTestWatcher.prototype.emitEvent = function(type, file, stat) {
  file = path.relative(this.root, file);

  if (type === DELETE_EVENT) {
    // Matching the non-polling API
    stat = null;
  }

  this.emit(type, file, this.root, stat);
  this.emit(ALL_EVENT, type, file, this.root, stat);
};

/**
 * End watching.
 *
 * @public
 */

PluginTestWatcher.prototype.close = function(callback) {
  Object.keys(this.watched).forEach(function(filepath) {
    fs.unwatchFile(filepath);
  });
  this.removeAllListeners();
  if (typeof callback === 'function') {
    setImmediate(callback.bind(null, null, true));
  }
};
