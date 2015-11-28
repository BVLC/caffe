'use strict';

var Promise = require('../ext/promise');
/*
 *
 * given an array of functions, that may or may not return promises sequence
 * will invoke them sequentially. If a promise is encountered sequence will
 * wait until it fulfills before moving to the next entry.
 *
 * ```js
 * var tasks = [
 *   function() { return Promise.resolve(1); },
 *   2,
 *   function() { return timeout(1000).then(function() { return 3; } },
 * ];
 *
 * sequence(tasks).then(function(results) {
 *   results === [
 *     1,
 *     2,
 *     3
 *   ]
 * });
 * ```
 *
 * @method sequence
 * @param tasks
 * @return Promise<Array>
 *
 */
module.exports = function sequence(tasks) {
  var length = tasks.length;
  var current = Promise.resolve();
  var results = new Array(length);

  for (var i = 0; i < length; ++i) {
    current = results[i] = current.then(tasks[i]);
  }

  return Promise.all(results);
};
