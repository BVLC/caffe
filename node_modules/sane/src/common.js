'use strict';

var minimatch = require('minimatch');

/**
 * Constants
 */

exports.DEFAULT_DELAY = 100;
exports.CHANGE_EVENT = 'change';
exports.DELETE_EVENT = 'delete';
exports.ADD_EVENT = 'add';
exports.ALL_EVENT = 'all';

/**
 * Assigns options to the watcher.
 *
 * @param {NodeWatcher|PollWatcher|WatchmanWatcher} watcher
 * @param {?object} opts
 * @return {boolean}
 * @public
 */

exports.assignOptions = function(watcher, opts) {
  opts = opts || {};
  watcher.globs = opts.glob || [];
  watcher.dot = opts.dot || false;
  if (!Array.isArray(watcher.globs)) {
    watcher.globs = [watcher.globs];
  }
  return opts;
};

/**
 * Checks a file relative path against the globs array.
 *
 * @param {array} globs
 * @param {string} relativePath
 * @return {boolean}
 * @public
 */

exports.isFileIncluded = function(globs, dot, relativePath) {
  var matched;
  if (globs.length) {
    for (var i = 0; i < globs.length; i++) {
      if (minimatch(relativePath, globs[i], {dot: dot})) {
        matched = true;
        break;
      }
    }
  } else {
    // Make sure we honor the dot option if even we're not using globs.
    if (!dot) {
      matched = minimatch(relativePath, '**/*', {dot: false});
    } else {
      matched = true;
    }
  }
  return matched;
};
