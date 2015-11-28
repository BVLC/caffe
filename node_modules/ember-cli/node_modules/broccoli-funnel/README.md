# Broccoli Funnel

[![Build Status](https://travis-ci.org/broccolijs/broccoli-funnel.svg?branch=master)](https://travis-ci.org/broccolijs/broccoli-funnel)
[![Build status](https://ci.appveyor.com/api/projects/status/3y3wo7hipq6d0cbp/branch/master?svg=true)](https://ci.appveyor.com/project/embercli/broccoli-funnel/branch/master)

Given an input node, the Broccoli Funnel plugin returns a new node with only a
subset of the files from the input node. The files can be moved to different
paths. You can use regular expressions to select which files to include or
exclude.

## Documentation

### `new Funnel(inputNode, options)`

`inputNode` *{Single node}*

A Broccoli node (formerly: "tree"). A node in Broccoli can be either a string
that references a directory in your project or a node object returned from
running another Broccoli plugin.

If your project has the following file structure:

```shell
.
├── Brocfile.js
└── src/
    ├── css/
    │   ├── reset.css
    │   └── todos.css
    ├── icons/
    │   ├── check-mark.png
    │   └── logo.jpg
    └── javascript/
        ├── app.js
        └── todo.js
```

You can select a subsection of the tree via Funnel:

```javascript
var Funnel = require('broccoli-funnel');
var cssFiles = new Funnel('src/css');

/*
  cssFiles contains the following files:

  ├── reset.css
  └── todos.css
*/

// export the node for Broccoli to begin processing
module.exports = cssFiles;
```

#### Options

`srcDir` *{String}*

A string representing the portion of the input node to start the funneling
from. This will be the base path for any `include`/`exclude` regexps.

Default: `'.'`, the root path of the input node.

If your project has the following file structure:

```shell
.
├── Brocfile.js
└── src/
    ├── css/
    │   ├── reset.css
    │   └── todos.css
    ├── icons/
    │   ├── check-mark.png
    │   └── logo.jpg
    └── javascript/
        ├── app.js
        └── todo.js
```

You can select a subsection of the node via Funnel:

```javascript
var Funnel = require('broccoli-funnel');
var MergeTrees = require('broccoli-merge-trees');

// root of our source files
var projectFiles = 'src';

/* get a new node of only files in the 'src/css' directory
  cssFiles contains the following files:

  ├── reset.css
  └── todos.css
*/
var cssFiles = new Funnel(projectFiles, {
  srcDir: 'css'
});

/* get a new node of only files in the 'src/icons' directory
  imageFiles contains the following files:

  ├── check-mark.png
  └── logo.jpg
*/
var imageFiles = new Funnel(projectFiles, {
  srcDir: 'icons'
});


module.exports = new MergeTrees([cssFiles, imageFiles]);
```

----

`destDir` *{String}*

A string representing the destination path that filtered files will be copied to.

Default: `'.'`, the root path of input node.

If your project has the following file structure:

```shell
.
├── Brocfile.js
└── src/
    ├── css/
    │   ├── reset.css
    │   └── todos.css
    ├── icons/
    │   ├── check-mark.png
    │   └── logo.jpg
    └── javascript/
        ├── app.js
        └── todo.js
```

You can select a subsection of the directory structure via Funnel and copy it to a new location:

```javascript
var Funnel = require('broccoli-funnel');

var cssFiles = new Funnel('src/css', {
  destDir: 'build'
});

/*
  cssFiles contains the following files:

  build/
  ├── reset.css
  └── todos.css
*/

module.exports = cssFiles;
```

----

`allowEmpty` *{Boolean}*

When using `srcDir`/`destDir` options only (aka no filtering via `include`/`exclude` options), if the `srcDir` were missing an error would be thrown.
Setting `allowEmpty` to true, will prevent that error by creating an empty directory at the destination path.

----

`include` *{Array of GlobStrings|RegExps|Functions}*

One or more matcher expression (regular expression, glob string, or function). Files within the node whose names match this
expression will be copied (with the location inside their parent directories
preserved) to the `destDir`.

Default: `[]`.

If your project has the following file structure

```shell
.
├── Brocfile.js
└── src/
    ├── css/
    │   ├── reset.css
    │   └── todos.css
    ├── icons/
    │   ├── check-mark.png
    │   └── logo.jpg
    └── javascript/
        ├── app.js
        └── todo.js
```

You can select files that match a glob expression and copy those subdirectories to a
new location, preserving their location within parent directories:

```javascript
var Funnel = require('broccoli-funnel');

// finds all files that match /todo/ and moves them
// the destDir
var todoRelatedFiles = new Funnel('src', {
  include: ['todo/**/*']
});

/*
  todoRelatedFiles contains the following files:
  .
  ├── css
  │   └── todos.css
  └── javascript
      └── todo.js
*/

module.exports = todoRelatedFiles;
```

----

`exclude` *{Array of Glob Strings|Glob Strings|Functions}*

One or more matcher expression (regular expression, glob string, or function). Files within the node whose names match this
expression will _not_ be copied to the `destDir` if they otherwise would have
been.

*Note, in the case when a file matches both an include and exclude pattern,
the exclude pattern wins*

Default: `[]`.

If your project has the following file structure:

```shell
.
├── Brocfile.js
└── src/
    ├── css/
    │   ├── reset.css
    │   └── todos.css
    ├── icons/
    │   ├── check-mark.png
    │   └── logo.jpg
    └── javascript/
        ├── app.js
        └── todo.js
```

You can select files that match a glob expression and exclude them from copying:

```javascript
var Funnel = require('broccoli-funnel');

// finds all files in 'src' EXCEPT those that match /todo/
// and adds them to a node.
var nobodyLikesTodosAnyway = new Funnel('src', {
  exclude: ['todo/**/*']
});

/*
  nobodyLikesTodosAnyway contains the following files:
  .
  ├── css
  │   └── reset.css
  ├── icons
  │   ├── check-mark.png
  │   └── logo.jpg
  └── javascript
      └── app.js
*/

module.exports = nobodyLikesTodosAnyway;
```

----

`files` *{Array of Strings}*

One or more relative file paths. Files within the node whose relative paths match
will be copied (with the location inside their parent directories
preserved) to the `destDir`.

Default: `[]`.

If your project has the following file structure

```shell
.
├── Brocfile.js
└── src/
    ├── css/
    │   ├── reset.css
    │   └── todos.css
    ├── icons/
    │   ├── check-mark.png
    │   └── logo.jpg
    └── javascript/
        ├── app.js
        └── todo.js
```

You can select a specific list of files copy those subdirectories to a
new location, preserving their location within parent directories:

```javascript
var Funnel = require('broccoli-funnel');

// finds these specific files and moves them to the destDir
var someFiles = new Funnel('src', {
  files: ['css/reset.css', 'icons/check-mark.png']
});

/*
  someFiles contains the following files:
  .
  ├── css
  │   └── reset.css
  └── icons
      └── check-mark.png
*/

module.exports = someFiles;
```

----

`getDestinationPath` *{Function}*

This method will get called for each file, receiving the currently processing
`relativePath` as its first argument. The value returned from
`getDestinationPath` will be used as the destination for the new node. This is
a very simple way to move files from one path to another (replacing the need
for `broccoli-file-mover` for example).

The return value of this method is cached for each input file. This means that
`getDestinationPath` will only be called once per `relativePath`.

In the following example, `getDestinationPath` is used to move `main.js` to
`ember-metal.js`:

```javascript
var node = new Funnel('packages/ember-metal/lib', {
  destDir: 'ember-metal',

  getDestinationPath: function(relativePath) {
    if (relativePath === 'lib/main.js') {
      return 'ember-metal.js';
    }

    return relativePath;
  }
});
```

## ZOMG!!! TESTS?!?!!?

I know, right?

Running the tests:

```javascript
npm install
npm test
```

## License

This project is distributed under the MIT license.
