This package contains tools for helping you write [transforms](https://github.com/substack/node-browserify#btransformtr) for [browserify](https://github.com/substack/node-browserify).

Many different transforms perform certain basic functionality, such as turning the contents of a stream into a string, or loading configuration from package.json.  This package contains helper methods to perform these common tasks, so you don't have to write them over and over again:

* `makeStringTransform()` creates a transform which consumes and returns a string, instead of using a stream.
* `makeFalafelTransform()` parses a JS file using [falafel](https://github.com/substack/node-falafel) and allows you to modify the code.
* `makeRequireTransform()` passes you the contents of each `require()` call in each script, and allows you to rewrite the require statement.
* All of the above will automatically search for transform configuration in package.json and pass it to you if available, but if you have a more complicated use case than the `make*Transform()` functions will support, then `loadTransformConfig()` will load configuration for you.
* `runTransform()` can be used to unit test your shiny new transform.


Installation
============

Install with `npm install --save browserify-transform-tools`.

Creating a String Transform
===========================
Browserify transforms work on streams.  This is all well and good, until you want to call a library like "falafel" which doesn't work with streams.  (If you're using falafel specifically, see below for `makeFalafelTransform`.)

Suppose you are writing a transform called "unbluify" which replaces all occurances of "blue" with a color loaded from a configuration:

```JavaScript
var options = {excludeExtensions: [".json"]};
module.exports = transformTools.makeStringTransform("unbluify", options,
    function (content, transformOptions, done) {
        var file = transformOptions.file;
        if(!transformOptions.config) {
            return done(new Error("Could not find unbluify configuration."));
        }

        done(null, content.replace(/blue/g, transformOptions.config.newColor));
    });
```

Notice that the color we replace "blue" with gets loaded from configuration.  The configuration
can be set in a variety of ways.  A simple example is to set it directly in package.json:

```JavaScript
{
    "name": "myProject",
    "version": "1.0.0",
    ...
    "unbluify": {"newColor": "red"}
}
```

See the section on "Loading Configuration" below for details on where configuration can be loaded from.

Parameters for `makeStringTransform()`:

* `transformFn(contents, transformOptions, done)` - Function which is called to
  do the transform.  `contents` are the contents of the file.  `done(err, transformed)` is
  a callback which must be called, passing the a string with the transformed contents of the
  file.  `transformOptions` consists of:

  * `transformOptions.file` is the name of the file (as would be passed to a normal browserify transform.)

  * `transformOptions.config` is the configuration for your transform, loaded either from
    browserify or from package.json.

  * `transformOptions.configData` is the configuration data for the transform (see
  `loadTransformConfig` below for details on where this comes from.)

* `options.excludeExtensions` - A list of extensions which will not be processed.  e.g.
  "['.coffee', '.jade']"

* `options.includeExtensions` - A list of extensions to process.  If this options is not
  specified, then all extensions will be processed.  If this option is specified, then
  any file with an extension not in this list will skipped.

* `options.jsFilesOnly` - If set true, then your transform will only run on "javascript" files.
  This is handy for Falafel and Require transforms, defined below.  This is equivalent to
  passing
  `includeExtensions: [".js", ".coffee", ".coffee.md", ".litcoffee", "._js", "._coffee"]`.

Creating a Falafel Transform
============================
Many transforms are based on [falafel](https://github.com/substack/node-falafel). browserify-transform-tools provides an easy way to define such transforms.  Here is an example which wraps all array expressions in a call to `fn()`:

```JavaScript
var options = {};
// Wraps all array expressions in a call to fn().  e.g. '[1,2,3]' becomes 'fn([1,2,3])'.
module.exports = transformTools.makeFalafelTransform("array-fnify", options,
    function (node, transformOptions, done) {
        if (node.type === 'ArrayExpression') {
            node.update('fn(' + node.source() + ')');
        }
        done();
    });
```

`makeFalafelTransform()` will be called once for every node in your JS file.  You can update the node.  Be sure to pass errors back via `done(err)`, and call `done()` when complete.

Options passed to `makeFalafelTransform()` are the same as for `makeStringTransform()`, as are the transformOptions passed to the transform function.  You can additionally pass a `options.falafelOptions` to `makeFalafelTransform` - this object will be passed as an options object directly to falafel.

Creating a Require Transform
============================

Many transforms are focused on transforming `require()` calls.  browserify-transform-tools has a solution for this:

```JavaScript
transform = transformTools.makeRequireTransform("requireTransform",
    {evaluateArguments: true},
    function(args, opts, cb) {
        if (args[0] === "foo") {
            return cb(null, "require('bar')");
        } else {
            return cb();
        }
    });
```

This will take all calls to `require("foo")` and transform them to `require('bar')`.  Note that makeRequireTransform can parse many simple expressions, so the above would succesfully parse `require("f" + "oo")`, for example.  Any expression involving core JavaScript, `__filename`, `__dirname`, `path`, and `join` (where join is an alias for `path.join`) can be parsed.  Setting the `evaluateArguments` option to false will disable this behavior, in which case the source code for everything inside the ()s will be returned.

Note that `makeRequireTransform` expects your function to return the complete `require(...)` call.  This makes it possible to write require transforms which will, for example, inline resources.

Again, all other options you can pass to `makeStringTransform` are valid here, too.

Loading Configuration
=====================

All `make*Transform()` functions will automatically load configuration for your transform and make it available via `transformOptions.config` (and through the more detailed `transformOptions.configData`.)  Functions are also provided for reading configuration if you are not using one of the `make*Transform()` functions.

Transform configuration can be loaded from a project's package.json file, from a js or coffee file specified in package.json, or programatically.  For details, see [the transform configuration documentation](https://github.com/benbria/browserify-transform-tools/wiki/Transform-Configuration).

Running a Transform
===================
If you want to unit test your transform, then `runTransform()` is for you:

```JavaScript
var myTransform = transformTools.makeFalafelTransform(...);
var dummyJsFile = path.resolve(__dirname, "../testFixtures/testWithConfig/dummy.js");
var content = "console.log('Hello World!');";
transformTools.runTransform(myTransform, dummyJsFile, {content: content},
    function(err, transformed) {
        // Verify transformed is what we expect...
    }
);
```

Thanks
======
Some of this was heavily inspired by:

* [ForbesLindesay](https://github.com/ForbesLindesay)'s [rfileify](https://github.com/ForbesLindesay/rfileify)
* [thlorenz](https://github.com/thlorenz)'s [browserify-shim](https://github.com/thlorenz/browserify-shim)

