## master

## 2.2.0

* add plugin.listEntries() – this returns a stat entry result, allowing
  subclasses access to the underyling stat information

## 2.1.0

* Performance improvements

## 2.0.3

* Fix bug

## 2.0.2

* Performance improvements

## 2.0.1

* Performance improvements

## 2.0.0

* Derive from broccoli-plugin base class, and expose same interface. In particular:

    * `updateCache(srcDirs, destDir)` becomes `build()`
    * We no longer derive from CoreObject
    * We gain the `name`, `annotation`, and `persistentOutput` options
    * `options` no longer auto-assigns to `this`; unknown options are ignored

* `filterFromCache.include`/`filterFromCache.exclude` are now called
  `cacheInclude`/`cacheExclude`; they must now be passed in through
  `options`, and can no longer be set on the prototype/instance

* Remove `enforceSingleInputTree` option; we now always expect an array

## 1.0.0

* Bump without changes

## 0.6.2

* Improve logging

## 0.6.1

* Ignore changes in directory size/mtime, so that excluded files can be added
  or removed without causing invalidations

## 0.6.0

* Use new [`.rebuild` API](https://github.com/broccolijs/broccoli/blob/master/docs/new-rebuild-api.md)

## 0.5.5

* Add ability to debug which files are causing an invalidation of the cache. The following will generate output indicating which path changed:

```
DEBUG=broccoli-caching-writer broccoli serve # if using broccoli-cli
```

## 0.5.4

* Update to newer core-object version.

## 0.5.3

* Ensure that errors are not thrown if `_destDir` has not been setup yet.

## 0.5.2

* Use `_destDir` for tracking the internal destination directory. This prevents collisions if inheritors use the common `destDir`
  name as a property.

## 0.5.1

* Allow easy inheritance. In your package's `index.js`:

```javascript
var CachingWriter = require('broccoli-caching-writer');

module.exports = CachingWriter.extend({
  init: function(inputTrees, options) {
    /* do additional setup here */
  },

  updateCache: function(srcPaths, destDir) {
    /* do main processing */
  }
});
```

Then in a consuming Brocfile:

```javascript
var MyFoo = require('my-foo'); // package from above

var tree = new MyFoo([someInput], { some: 'options' });
```

## 0.5.0

* Allow filtering on files to include/exclude when determining when to invalidate the cache. This allows
  you to use simple regular expressions to prevent invalidating the cache when files that do not affect the
  tree in question are changed.

```javascript
var outputTree = compileCompass(inputTree, {
  filterFromCache: {
    include: [
      /.(scss|sass)$/   // only base the input tree’s hash on *.scss and *.sass files
    ]
  }
});
```

  This does _not_ affect what files make it to the output tree at all, rather it only makes it easier
  for subclasses to only rebuild when file types they care about change.

* Symlink from cache instead of manually hard-linking. This should be a speed improvement
  for posix platforms, and will soon be able to take advantage of improvements for Windows
  (for those curious stay tuned on Windows support [here](https://github.com/broccolijs/node-symlink-or-copy/pull/1)).

* Allow multiple input trees. If an array of trees is passed to the constructor, all trees will be read and their collective
  output will be used to calculate the cache (any trees invalidating causes `updateCache` to be called).

  The default now is to assume that an array of trees is allowed, if you want to opt-out of this behavior set `enforceSingleInputTree`
  to `true` on your classes prototype.

  By default an array of paths will now be the first argument to `updateCache` (instead of a single path in prior versions). The
  `enforceSingleInputTree` property also controls this.

* Due to the changes above (much more being done in our constructor), inheritors are now required to call the `broccoli-caching-writer`
  constructor from their own.
