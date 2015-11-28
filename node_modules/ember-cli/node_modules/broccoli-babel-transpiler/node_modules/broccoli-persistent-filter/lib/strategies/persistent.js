'use strict';

var AsyncDiskCache = require('async-disk-cache');
var Promise = require('rsvp').Promise;

module.exports = {

  _cache: {},

  init: function(ctx) {
    if (!ctx.constructor._persistentCacheKey) {
      ctx.constructor._persistentCacheKey = this.cacheKey(ctx);
    }

    this._cache = new AsyncDiskCache(ctx.constructor._persistentCacheKey, {
      location: process.env['BROCCOLI_PERSISTENT_FILTER_CACHE_ROOT'],
      compression: 'deflate'
    });
  },

  cacheKey: function(ctx) {
    return ctx.cacheKey();
  },

  processString: function(ctx, contents, relativePath) {
    var key = ctx.cacheKeyProcessString(contents, relativePath);
    var cache = this._cache;

    return cache.get(key).then(function(entry) {
      if (entry.isCached) {
        ctx._debug('persistent cache hit: %s', relativePath);
        return entry.value;
      } else {
        ctx._debug('persistent cache prime: %s', relativePath);
        var string = Promise.resolve(ctx.processString(contents, relativePath));

        return string.then(function(string) {
          return cache.set(key, string).then(function() {
            return string;
          });
        });
      }
    });
  }
};
