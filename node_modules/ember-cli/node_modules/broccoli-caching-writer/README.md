# Broccoli Caching Writer

[![Build Status](https://travis-ci.org/ember-cli/broccoli-caching-writer.svg?branch=master)](https://travis-ci.org/ember-cli/broccoli-caching-writer)
[![Build status][appveyor-badge]][appveyor-badge-url]

Drop-in-replacement for
[broccoli-plugin](https://github.com/broccolijs/broccoli-plugin) adding a thin
caching layer based on the computed hash of the input directory trees. If any
file in an input node has changed, the `build` method will be called,
otherwise (if input is the same) the results of the last `build` call will be
used instead.

## Example

Say your plugin derives from `Plugin` like so:

```js
var Plugin = require('broccoli-plugin');

MyPlugin.prototype = Object.create(Plugin.prototype);
MyPlugin.prototype.constructor = MyPlugin;
function MyPlugin(inputNodes, options) {
  options = options || {};
  Plugin.call(this, inputNodes, {
    annotation: options.annotation
  });
}
```

To add caching, simply replace `Plugin` with `CachingWriter`, like so:

```js
var CachingWriter = require('broccoli-caching-writer');

MyPlugin.prototype = Object.create(CachingWriter.prototype);
MyPlugin.prototype.constructor = MyPlugin;
function MyPlugin(inputNodes, options) {
  options = options || {};
  CachingWriter.call(this, inputNodes, {
    annotation: options.annotation
  });
}
```


## Documentation

### `new CachingWriter(inputNodes, options)`

Call this base class constructor from your subclass constructor.

* `inputNodes`: An array of input nodes.

* `options`:

    * `name`, `annotation`, `persistentOutput`: Same as
      [broccoli-plugin](https://github.com/broccolijs/broccoli-plugin#new-plugininputnodes-options);
      see there.

    * `cacheInclude` (default: `[]`): An array of regular expressions that files and directories in an input node must pass (match at least one pattern) in order to be included in the cache hash for rebuilds. In other words, a whitelist of patterns that identify which files and/or directories can trigger a rebuild.

    * `cacheExclude` (default: `[]`): An array of regular expressions that files and directories in an input node cannot pass in order to be included in the cache hash for rebuilds. In other words, a blacklist of patterns that identify which files and/or directories will never trigger a rebuild.

        *Note, in the case when a file or directory matches both an include and exlude pattern, the exclude pattern wins*


## ZOMG!!! TESTS?!?!!?

I know, right?

Running the tests:

```javascript
npm install
npm test
```

## License

This project is distributed under the MIT license.

[appveyor-badge]: https://ci.appveyor.com/api/projects/status/ocfp2hqo7hyhyy80?svg=true
[appveyor-badge-url]: https://ci.appveyor.com/project/embercli/broccoli-caching-writer/branch/master
