'use strict';

/**
 * Cache results of the first function call to ensure only calling once.
 *
 * ```js
 * var lazy = require('lazy-cache')(require);
 * // cache the call to `require('ansi-yellow')`
 * lazy('ansi-yellow', 'yellow');
 * // use `ansi-yellow`
 * console.log(lazy.yellow('this is yellow'));
 * ```
 *
 * @param  {Function} `fn` Function that will be called only once.
 * @return {Function} Function that can be called to get the cached function
 * @api public
 */

function lazyCache(fn) {
  var cache = {};
  var proxy = function (mod, name) {
    name = name || camelcase(mod);
    Object.defineProperty(proxy, name, {
      get: getter
    });

    function getter () {
      if (cache.hasOwnProperty(name)) {
        return cache[name];
      }
      try {
        return (cache[name] = fn(mod));
      } catch (err) {
        err.message = 'lazy-cache ' + err.message + ' ' + __filename;
        throw err;
      }
    }
    return getter;
  };
  return proxy;
}

/**
 * Used to camelcase the name to be stored on the `lazy` object.
 *
 * @param  {String} `str` String containing `_`, `.`, `-` or whitespace that will be camelcased.
 * @return {String} camelcased string.
 */

function camelcase(str) {
  if (str.length === 1) { return str.toLowerCase(); }
  str = str.replace(/^[\W_]+|[\W_]+$/g, '').toLowerCase();
  return str.replace(/[\W_]+(\w|$)/g, function (_, ch) {
    return ch.toUpperCase();
  });
}

/**
 * Expose `lazyCache`
 */

module.exports = lazyCache;
