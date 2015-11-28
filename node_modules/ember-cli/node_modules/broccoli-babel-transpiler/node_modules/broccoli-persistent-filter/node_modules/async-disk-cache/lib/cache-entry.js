'use strict';

function CacheEntry(isCached, key, value) {
  this.isCached = isCached;
  this.key = key;
  this.value = value;
}

module.exports = CacheEntry;
module.exports.MISS = new CacheEntry(false);

