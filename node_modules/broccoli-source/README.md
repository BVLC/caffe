# Broccoli Source

[![Build Status](https://travis-ci.org/broccolijs/broccoli-source.svg?branch=master)](https://travis-ci.org/broccolijs/broccoli-source)

Broccoli plugin for creating "source" nodes that refer to physical file system
directories.

## Example Usage

In `Brocfile.js`:

```js
var broccoliSource = require('broccoli-source');
var WatchedDir = broccoliSource.WatchedDir;
var UnwatchedDir = broccoliSource.UnwatchedDir;

// Refers to the ./lib directory on disk, and watches it.
var lib = new WatchedDir('lib');
// Note: this is equivalent to the deprecated plain-string syntax:
//var lib = 'lib';

// Refers to the ./bower_components/jquery directory, but does not watch it.
var jquery = new UnwatchedDir('bower_components/jquery');
```

## Reference

### `new WatchedDir(directoryPath, options)`

Create a Broccoli node referring to a directory on disk. The Broccoli watcher
used by `broccoli serve` will watch the directory and all subdirectories, and
trigger a rebuild whenever something changes.

* `directoryPath`: A path to a directory, either absolute, or relative to the
  working directory (typically the directory containing `Brocfile.js`).

  The directory must exist, or Broccoli will abort.

* `options`

     * `annotation`: A human-readable description for this node.

### `new UnwatchedDir(directoryPath, options)`

Same as `WatchedDir`, but the directory will not be watched.

This can be useful for performance reasons. For example, say you want to refer
to a large directory hierarchy of third-party code in your `Brocfile.js`. Such
third-party code is rarely edited in practice. Using `UnwatchedDir` instead of
`WatchedDir` saves the overhead of setting up useless file system watchers.

When in doubt, use `WatchedDir` instead.
