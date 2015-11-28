# Broccoli

<img src="logo/broccoli-logo.generated.png" align="right" height="150">

[![Build Status](https://travis-ci.org/broccolijs/broccoli.svg?branch=master)](https://travis-ci.org/broccolijs/broccoli)
[![Build status](https://ci.appveyor.com/api/projects/status/jd3ts93gryjeqclf/branch/master?svg=true)](https://ci.appveyor.com/project/joliss/broccoli/branch/master)

A fast, reliable asset pipeline, supporting constant-time rebuilds and compact
build definitions. Comparable to the Rails asset pipeline in scope, though it
runs on Node and is backend-agnostic. For background and architecture, see the
[introductory blog post](http://www.solitr.com/blog/2014/02/broccoli-first-release/).

For the command line interface, see
[broccoli-cli](https://github.com/broccolijs/broccoli-cli).

**This is 0.x beta software.**

Windows support is still spotty. Our biggest pain point is unreliable file
deletion (see [rimraf#72](https://github.com/isaacs/rimraf/issues/72)).

## Installation

```bash
npm install --save-dev broccoli
npm install --global broccoli-cli
```

## Brocfile.js

A `Brocfile.js` file in the project root contains the build specification. It
should export a tree.

A tree can be any string representing a directory path, like `'app'` or
`'src'`. Or a tree can be an object conforming to the [Plugin API
Specification](#plugin-api-specification). A `Brocfile.js` will usually
directly work with only directory paths, and then use the plugins in the
[Plugins](#plugins) section to generate transformed trees.

The following simple `Brocfile.js` would export the `app/` subdirectory as a
tree:

```js
module.exports = 'app'
```

With that Brocfile, the build result would equal the contents of the `app`
tree in your project folder. For example, say your project contains these
files:

    app
    ├─ main.js
    └─ helper.js
    Brocfile.js
    package.json
    …

Running `broccoli build the-output` (a command provided by
[broccoli-cli](https://github.com/broccolijs/broccoli-cli)) would generate
the following folder within your project folder:

    the-output
    ├─ main.js
    └─ helper.js

### Using plugins in a `Brocfile.js`

The following `Brocfile.js` exports the `app/` subdirectory as `appkit/`:

```js
var Funnel = require('broccoli-funnel')

module.exports = new Funnel('app', {
  destDir: 'appkit'
})
```

That example uses the plugin
[`broccoli-funnel`](https://www.npmjs.com/package/broccoli-funnel).
In order for the `require` call to work, you must first put the plugin in
your `devDependencies` and install it, with

    npm install --save-dev broccoli-funnel

With the above `Brocfile.js` and the file tree from the previous example,
running `broccoli build the-output` would generate the following folder:

    the-output
    └─ appkit
       ├─ main.js
       └─ helper.js

### A larger example

You can see a full-featured `Brocfile.js` in
[broccoli-sample-app](https://github.com/broccolijs/broccoli-sample-app/blob/master/Brocfile.js).

## Plugins

You can find plugins on [broccoliplugins.com](http://broccoliplugins.com) or under the [broccoli-plugin keyword](https://www.npmjs.org/browse/keyword/broccoli-plugin) on npm.

### Running Broccoli, Directly or Through Other Tools

* [broccoli-timepiece](https://github.com/rjackson/broccoli-timepiece)
* [grunt-broccoli](https://github.com/quandl/grunt-broccoli)
* [grunt-broccoli-build](https://github.com/ericf/grunt-broccoli-build)

### Helpers

Shared code for writing plugins.

* [broccoli-caching-writer](https://github.com/rjackson/broccoli-caching-writer)
* [broccoli-filter](https://github.com/broccolijs/broccoli-filter)
* [broccoli-writer](https://github.com/broccolijs/broccoli-writer)
* [node-quick-temp](https://github.com/joliss/node-quick-temp)

## Plugin API Specification

Broccoli defines a single plugin API: a tree. A tree object represents a tree
(directory hierarchy) of files that will be regenerated on each build.

By convention, plugins will export a function that takes one or more input
trees, and returns an output tree object. Usually your plugin will be
implemented as a class representing a tree, but it is recommended to make the
`new` operator optional
([example](https://github.com/joliss/broccoli-coffee/blob/a55b3a6677f6d9da83334e9c916ae5e57895d1a6/index.js#L8)).

A tree object must supply two methods that will be called by Broccoli:

### `tree.read(readTree)`

The `.read` method must return a path or a promise for a path, containing the
tree contents.

It receives a `readTree` function argument from Broccoli. If `.read` needs to
read other trees, it must not call `otherTree.read` directly. Instead, it must
call `readTree(otherTree)`, which returns a promise for the path containing
`otherTree`'s contents. It must not call `readTree` again until the promise
has resolved; that is, it cannot call `readTree` on multiple trees in
parallel.

Broccoli will call the `.read` method repeatedly to rebuild the tree, but at
most once per rebuild; that is, if a tree is used multiple times in a build
definition, Broccoli will reuse the path returned instead of calling `.read`
again.

The `.read` method is responsible for creating a new temporary directory to
store the tree contents in. Subsequent invocations of `.read` should remove
temporary directories created in previous invocations.

### `tree.cleanup()`

For every tree whose `.read` method was called one or more times, the
`.cleanup` method will be called exactly once. No further `.read` calls will
follow `.cleanup`. The `.cleanup` method should remove all temporary
directories created by `.read`.

### Debugging


#### Errors

When it is known which file caused a given error, plugin authors can make errors
easier to track down by setting the `.file` property on the generated error.

This `.file` property is used by both the console logging, and the server middleware
to display more helpful error messages.

#### Descriptive Naming

As of 0.11 Broccoli prints a log of any trees that took a significant amount of the total
build time to assist in finding which trees are consuming the largest build times.

To determine the name to be printed Broccoli will first look for a `.description`
property on the plugin instance then fall back to using the plugin constructor's name.

## Security

* Do not run `broccoli serve` on a production server. While this is
  theoretically safe, it exposes a needlessly large amount of attack surface
  just for serving static assets. Instead, use `broccoli build` to precompile
  your assets, and serve the static files from a web server of your choice.

## Get Help

* IRC: `#broccolijs` on Freenode. Ask your question and stick around for a few
  hours. Someone will see your message eventually.
* Twitter: mention @jo_liss with your question
* GitHub: Open an issue on a specific plugin repository, or on this
  repository for general questions.

## License

Broccoli was originally written by [Jo Liss](http://www.solitr.com/) and is
licensed under the [MIT license](LICENSE).

The Broccoli logo was created by [Samantha Penner
(Miric)](http://mirics.deviantart.com/) and is licensed under [CC0
1.0](https://creativecommons.org/publicdomain/zero/1.0/).
