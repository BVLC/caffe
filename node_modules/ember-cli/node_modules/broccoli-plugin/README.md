# The Broccoli Plugin Base Class

[![Build Status](https://travis-ci.org/broccolijs/broccoli-plugin.svg?branch=master)](https://travis-ci.org/broccolijs/broccoli-plugin)
[![Build status](https://ci.appveyor.com/api/projects/status/k4tk8b99m1e58ftd?svg=true)](https://ci.appveyor.com/project/joliss/broccoli-plugin)

## Example Usage

```js
var Plugin = require('broccoli-plugin');
var path = require('path');

// Create a subclass MyPlugin derived from Plugin
MyPlugin.prototype = Object.create(Plugin.prototype);
MyPlugin.prototype.constructor = MyPlugin;
function MyPlugin(inputNodes, options) {
  options = options || {};
  Plugin.call(this, inputNodes, {
    annotation: options.annotation
  });
  this.options = options;
}

MyPlugin.prototype.build = function() {
  // Read files from this.inputPaths, and write files to this.outputPath.
  // Silly example:

  // Read 'foo.txt' from the third input node
  var inputBuffer = fs.readFileSync(path.join(this.inputPaths[2], 'foo.txt'));
  var outputBuffer = someCompiler(inputBuffer);
  // Write to 'bar.txt' in this node's output
  fs.writeFileSync(path.join(this.outputPath, 'bar.txt'), outputBuffer);
};
```

## Reference

### `new Plugin(inputNodes, options)`

Call this base class constructor from your subclass constructor.

* `inputNodes`: An array of node objects that this plugin will read from.
  Nodes are usually other plugin instances; they were formerly known as
  "trees".

* `options`

    * `name`: The name of this plugin class. Defaults to `this.constructor.name`.
    * `annotation`: A descriptive annotation. Useful for debugging, to tell
      multiple instances of the same plugin apart.
    * `persistentOutput`: If true, the output directory is not automatically
      emptied between builds.

### `Plugin.prototype.build()`

Override this method in your subclass. It will be called on each (re-)build.

This function will typically access the following read-only properties:

* `this.inputPaths`: An array of paths on disk corresponding to each node in
  `inputNodes`. Your plugin will read files from these paths.

* `this.outputPath`: The path on disk corresponding to this plugin instance
  (this node). Your plugin will write files to this path. This directory is
  emptied by Broccoli before each build, unless the `persistentOutput` options
  is true.

* `this.cachePath`: The path on disk to an auxiliary cache directory. Use this
  to store files that you want preserved between builds. This directory will
  only be deleted when Broccoli exits.

All paths stay the same between builds.

To perform asynchronous work, return a promise. The promise's eventual value
is ignored (typically `null`).

To report a compile error, `throw` it or return a rejected promise. Also see
section "Error Objects" below.

### `Plugin.prototype.getCallbackObject()`

Advanced usage only.

Return the object on which Broccoli will call `obj.build()`. Called once after
instantiation. By default, returns `this`. Plugins do not usually need to
override this, but it can be useful for base classes that other plugins in turn
derive from, such as
[broccoli-caching-writer](https://github.com/ember-cli/broccoli-caching-writer).

For example, to intercept `.build()` calls, you might
`return { build: this.buildWrapper.bind(this) }`.
Or, to hand off the plugin implementation to a completely separate object:
`return new MyPluginWorker(this.inputPaths, this.outputPath, this.cachePath)`,
where `MyPluginWorker` provides a `.build` method.

### Error Objects

To help with displaying clear error messages for build errors, error objects
may have the following optional properties in addition to the standard
`message` property:

* `file`: Path of the file in which the error occurred, relative to one of the
  `inputPaths` directories
* `treeDir`: The path that `file` is relative to. Must be an element of
  `this.inputPaths`. (The name `treeDir` is for historical reasons.)
* `line`: Line in which the error occurred (one-indexed)
* `column`: Column in which the error occurred (zero-indexed)
