# fs-tree-diff [![Build Status](https://travis-ci.org/stefanpenner/fs-tree-diff.svg)](https://travis-ci.org/stefanpenner/fs-tree-diff)

FSTree provides the means to calculate a patch (set of operations) between one file system tree and another.

The possible operations are:

* `unlink` – remove the specified file
* `rmdir` – remove the specified folder
* `mkdir` – create the specified folder
* `create` – create the specified file
* `update` – update the specified file

The operations choosen aim to minimize the amount of IO required to apply a given patch.
For example, a naive `rm -rf` of a directory tree is actually quite costly, as child directories
must be recursively traversed, entries stated.. etc, all to figure out what first must be deleted.
Since we patch from tree to tree, discovering new files is both wasteful and un-needed.

The operations will also be provided in the correct order. So when deleting a large tree, unlink
and rmdir operations will be provided depthFirst. Allowing us to safely replay the operations without having to first confirm the FS is as we expected.

A simple example:

```js
var FSTree = require('fs-tree-diff');
var current = FSTree.fromPaths([
  'a.js'
]);

var next = FSTree.fromPaths({
  'b.js'
});

current.calculatePatch(next) === [
  ['unlink', 'a.js'],
  ['create', 'b.js']
];
```

A slightly more complicated example:

```js
var FSTree = require('fs-tree-diff');
var current = FSTree.fromPaths([
  'a.js',
  'b/f.js'
]);

var next = FSTree.fromPaths({
  'b.js',
  'b/c/d.js'
  'b/e.js'
});

current.calculatePatch(next) === [
  ['unlink', 'a.js'],
  ['unlink', 'b/f.js'],
  ['create', 'b.js'],
  ['mkdir', 'b/c'],
  ['create', 'b/c/d.js'],
  ['create', 'b/e.js']
];
```

Now, the above examples do not demonstrate `update` operations. This is because when providing only paths, we do not have sufficient information to check if one entry is merely different from another with the same relativePath.

For this, FSTree supports more complex input structure. To demonstrate, We will use the [walk-sync](https://github.com/joliss/node-walk-sync) module. Which provides higher fidelity input, allowing FSTree to also detect changes. More on what an [entry from walkSync.entries is](https://github.com/joliss/node-walk-sync#entries)

```js
var walkSync = require('walk-sync');

// path/to/root/foo.js
// path/to/root/bar.js
var current = new FSTree({
  entries: walkSync.entries('path/to/root')
});

writeFileSync('path/to/root/foo.js', 'new content');
writeFileSync('path/to/root/baz.js', 'new file');

var next = new FSTree({
  entries: walkSync.entries('path/to/root')
});

current.calculatePatch(next) === [
  ['update', 'foo.js'], // mtime + size changed, so this input is stale and needs updating.
  ['create', 'baz.js']  // new file, so we should create it
  /* bar stays the same and is left inert*/
];

```
