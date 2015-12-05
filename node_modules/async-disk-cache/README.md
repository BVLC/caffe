# async-disk-cache [![Build status](https://ci.appveyor.com/api/projects/status/lfliompah66m611x?svg=true)](https://ci.appveyor.com/project/embercli/async-disk-cache) [![Build Status](https://travis-ci.org/stefanpenner/async-disk-cache.svg)](https://travis-ci.org/stefanpenner/async-disk-cache)


An aync disk cache. inspired by [jgable/cache-swap](https://github.com/jgable/cache-swap)

A sync sibling version is also available: [stefanpenner/sync-disk-cache](https://github.com/stefanpenner/sync-disk-cache/)

## Example

```js
var Cache = require('async-disk-cache');
var cache = new Cache('my-cache');
// 'my-cache' also serves as the global key for the cache.
// if you have multiple programs with this same `cache-key` they will share the
// same backing store. This by design.

// checking
cache.has('foo').then(function(wasFooFound) {

});

// retrieving (cache hit)
cache.get('foo').then(function(cacheEntry) {
  cacheEntry === {
    isCached: true,
    path: 'foo',
    content: 'content of foo'
  }
});

// retrieving (cache miss)
cache.get('foo').then(function(cacheEntry) {
  cacheEntry === {
    isCached: false,
    path: 'foo',
    content: undefined
  }
});

// retrieving (cache miss)
cache.set('foo', 'content of foo').then(function() {
  // was set
});

// clearing the cache

cache.clear().then(function() {
  // cache was cleared
})
```


Enable compression:

```js
var Cache = require('sync-disk-cache');
var cache = new Cache('my-cache', {
  compression: 'gzip' | 'deflate' | 'deflateRaw' // basically just what nodes zlib's ships with
})
```

## License

Licensed under the MIT License, Copyright 2015 Stefan Penner
