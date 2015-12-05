# broccoli-writer

Base class for Broccoli plugins that write an output tree. Most plugins fall
into this category (the exception being plugins that just point at an existing
directory on the file system, like broccoli-bower), so they will be
implemented either using this base class or using a derived, more specific
base class.

This base class's main purpose is to create and clean up the temporary output
directory for you.

## Installation

```
npm --save broccoli-writer
```

## Usage

Write your plugin like so:

```js
var Writer = require('broccoli-writer');

module.exports = MyCompiler;
MyCompiler.prototype = Object.create(Writer.prototype);
MyCompiler.prototype.constructor = MyCompiler;
function MyCompiler (arg1, arg2, ...) {
  if (!(this instanceof MyCompiler)) return new MyCompiler(arg1, arg2, ...);
  ...
};

MyCompiler.prototype.write = function (readTree, destDir) {
  ...
};
```

Inside `MyCompiler.prototype.write`, `readTree` is [supplied by
Broccoli](https://github.com/joliss/broccoli#plugin-api-specification) -- call
`readTree(someInputTree)` to read another tree. `destDir` is the path to a
newly-created temporary directory created by the `Writer` base class. Place
all the output files you wish to generate in this directory.

If you want to do something asynchronous, return a promise that resolves when
you are done.

In the `MyCompiler` constructor, you don't need to call the `Writer` base
class constructor.

Your plugin can be used in `Brocfile.js` like so:

```js
var compileSomething = require('broccoli-my-compiler');

var outputTree = compileSomething(arg1, arg2, ...)
```
