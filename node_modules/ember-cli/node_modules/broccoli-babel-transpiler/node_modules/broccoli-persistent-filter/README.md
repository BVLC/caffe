# broccoli-persistent-filter

[![Build Status](https://travis-ci.org/stefanpenner/broccoli-persistent-filter.svg?branch=master)](https://travis-ci.org/stefanpenner/broccoli-persistent-filter)
[![Build status](https://ci.appveyor.com/api/projects/status/gvt0rheb1c2c4jwd/branch/master?svg=true)](https://ci.appveyor.com/project/embercli/broccoli-persistent-filter/branch/master)

Helper base class for Broccoli plugins that map input files into output files. Except with a persistent cache to fast restarts.
one-to-one.

## API

```js
class Filter {
  /**
   * Abstract base-class for filtering purposes.
   *
   * Enforces that it is invoked on an instance of a class which prototypically
   * inherits from Filter, and which is not itself Filter.
   */
  constructor(inputNode: BroccoliNode, options: FilterOptions): Filter;

  /**
   * Abstract method `processString`: must be implemented on subclasses of
   * Filter.
   *
   * The return value is written as the contents of the output file
   */
  abstract processString(contents: string, relativePath: string): string;

  /**
   * Virtual method `getDestFilePath`: determine whether the source file should
   * be processed, and optionally rename the output file when processing occurs.
   *
   * Return `null` to pass the file through without processing. Return
   * `relativePath` to process the file with `processString`. Return a
   * different path to process the file with `processString` and rename it.
   *
   * By default, if the options passed into the `Filter` constructor contain a
   * property `extensions`, and `targetExtension` is supplied, the first matching
   * extension in the list is replaced with the `targetExtension` option's value.
   */
  virtual getDestFilePath(relativePath: string): string;
}
```

### Options

* `extensions`: An array of file extensions to process, e.g. `['md', 'markdown']`.
* `targetExtension`: The file extension of the corresponding output files, e.g.
  `'html'`.
* `inputEncoding`: The character encoding used for reading input files to be
  processed (default: `'utf8'`). For binary files, pass `null` to receive a
  `Buffer` object in `processString`.
* `outputEncoding`: The character encoding used for writing output files after
  processing (default: `'utf8'`). For binary files, pass `null` and return a
  `Buffer` object from `processString`.
* `name`, `annotation`: Same as
  [broccoli-plugin](https://github.com/broccolijs/broccoli-plugin#new-plugininputnodes-options);
  see there.

All options except `name` and `annotation` can also be set on the prototype
instead of being passed into the constructor.

### Example Usage

```js
var Filter = require('broccoli-filter');

Awk.prototype = Object.create(Filter.prototype);
Awk.prototype.constructor = Awk;
function Awk(inputNode, search, replace, options) {
  options = options || {};
  Filter.call(this, inputNode, {
    annotation: options.annotation
  });
  this.search = search;
  this.replace = replace;
}

Awk.prototype.extensions = ['txt'];
Awk.prototype.targetExtension = 'txt';

Awk.prototype.processString = function(content, relativePath) {
  return content.replace(this.search, this.replace);
};
```

In `Brocfile.js`, use your new `Awk` plugin like so:

```
var node = new Awk('docs', 'ES6', 'ECMAScript 2015');

module.exports = node;
```

## Persistent Cache

Adding persist flag allows a subclass to persist state across restarts. This exists to mitigate the upfront cost of some more expensive transforms on warm boot. __It does not aim to improve incremental build performance, if it does, it should indicate something is wrong with the filter or input filter in question.__

### How does it work?

It does so but establishing a 2 layer file cache. The first layer, is the entire bucket.
The second, `cacheKeyProcessString` is a per file cache key.

Together, these two layers should provide the right balance of speed and sensibility.

The bucket level cacheKey must be stable but also never become stale. If the key is not
stable, state between restarts will be lost and performance will suffer. On the flip-side,
if the cacheKey becomes stale changes may not be correctly reflected.

It is configured by subclassing and refining `cacheKey` method. A good key here, is
likely the name of the plugin, its version and the actual versions of its dependencies.

```js
Subclass.prototype.cacheKey = function() {
 return md5(Filter.prototype.call(this) + inputOptionsChecksum + dependencyVersionChecksum);
}
```

The second key, represents the contents of the file. Typically the base-class's functionality
is sufficient, as it merely generates a checksum of the file contents. If for some reason this
is not sufficient, it can be re-configured via subclassing.

```js
Subclass.prototype.cacheKeyProcessString = function(string, relativePath) {
  return superAwesomeDigest(string);
}
```

It is recommended that persistent re-builds is opt-in by the consumer as it does not currently work on all systems.

```js
var myTree = new SomePlugin('lib', { persist: true });
```

## FAQ

### Upgrading from 0.1.x to 1.x

You must now call the base class constructor. For example:

```js
// broccoli-filter 0.1.x:
function MyPlugin(inputTree) {
  this.inputTree = inputTree;
}

// broccoli-filter 1.x:
function MyPlugin(inputNode) {
  Filter.call(this, inputNode);
}
```

Note that "node" is simply new terminology for "tree".

### Source Maps

**Can this help with compilers that are almost 1:1, like a minifier that takes
a `.js` and `.js.map` file and outputs a `.js` and `.js.map` file?**

Not at the moment. I don't know yet how to implement this and still have the
API look beautiful. We also have to make sure that caching works correctly, as
we have to invalidate if either the `.js` or the `.js.map` file changes. My
plan is to write a source-map-aware uglifier plugin to understand this use
case better, and then extract common code back into this `Filter` base class.
