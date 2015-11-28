# broccoli-babel-transpiler

[![Build Status](https://travis-ci.org/babel/broccoli-babel-transpiler.svg?branch=master)](https://travis-ci.org/babel/broccoli-babel-transpiler)
[![Build status](https://ci.appveyor.com/api/projects/status/a0nbd84m1x4y5fp5?svg=true)](https://ci.appveyor.com/project/embercli/broccoli-babel-transpiler)


A [Broccoli](https://github.com/broccolijs/broccoli) plugin which
transpiles ES6 to readable ES5 by using [babel](https://github.com/babel/babel).

## How to install?

```sh
$ npm install broccoli-babel-transpiler --save-dev
```

## How to use?

In your `Brocfile.js`:

```js
var esTranspiler = require('broccoli-babel-transpiler');
var scriptTree = esTranspiler(inputTree, options);
```

You can find [options](https://babeljs.io/docs/usage/options) at babel's
github repo.

### Examples

You'll find three example projects using this plugin in the repository [broccoli-babel-examples](https://github.com/givanse/broccoli-babel-examples).
Each one of them builds on top of the previous example so you can progess from bare minimum to ambitious development.

 * [es6-fruits](https://github.com/givanse/broccoli-babel-examples/tree/master/es6-fruits) - Execute a single ES6 script.
 * [es6-website](https://github.com/givanse/broccoli-babel-examples/tree/master/es6-website) - Build a simple website.
 * [es6-modules](https://github.com/givanse/broccoli-babel-examples/tree/master/es6-modules) - Handle modules and unit tests.
 
## About source map

Currently this plugin only supports inline source map. If you need
separate source map feature, you're welcome to submit a pull request.

## Advanced usage

`filterExtensions` is an option to limit (or expand) the set of file extensions that will be transformed.

The default `filterExtension` is `js`

```js
var esTranspiler = require('broccoli-babel-transpiler');
var scriptTree = esTranspiler(inputTree, {
    filterExtensions:['js', 'es6'] // babelize both .js and .es6 files
});
```

`exportModuleMetadata` is an option that can be used to write a JSON file to the output tree that gives you metadata about the tree's imports and exports.

## Polyfill

In order to use some of the ES6 features you must include the Babel [polyfill](http://babeljs.io/docs/usage/polyfill/#usage-in-browser).

You don't always need this, review which features need the polyfill here: [ES6 Features](https://babeljs.io/docs/learn-es6).

```js
var esTranspiler = require('broccoli-babel-transpiler');
var scriptTree = esTranspiler(inputTree, { browserPolyfill: true });
```
