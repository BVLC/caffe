# node-symlink-or-copy

Symlink a file or directory to another place. Fall back to copying on Windows.
Made for use with Broccoli plugins, for "do what I mean" behavior.

## Installation

```sh
npm install --save symlink-or-copy
```

## Example

```js
var symlinkOrCopySync = require('symlink-or-copy').sync;

symlinkOrCopySync('src_dir/some_file.txt', 'dest_dir/some_file.txt');
symlinkOrCopySync('src_dir/some_dir', 'dest_dir/some_dir');
```

## Description

```js
symlinkOrCopySync(srcPath, destPath)
```

Create a symlink at `destPath` pointing to `srcPath`.

On Windows, we may fall back to copying `srcPath` to `destPath`, preserving
last-modified times. However, do not *rely* on always getting a copy on
Windows (see Notes below).

If you pass a relative `srcPath`, it will be resolved relative to
`process.cwd()`, akin to a copy function. Note that this is unlike
[`fs.symlinkSync`](http://nodejs.org/api/fs.html#fs_fs_symlink_srcpath_dstpath_type_callback),
whose `srcPath` is relative to `destPath`.

If `srcPath` does not exist or is a broken symlink, we might throw an
exception, or we might create a broken symlink.

When we fall back to copying, symlinks at or beneath `srcPath` will be
dereferenced, and broken symlinks will cause exceptions.

We will throw an exception if `destPath` already exists. Thus in contrast to
Unix `cp` or `ln`, the following will fail:

```js
// dest_dir already exists, and we might expect dest_dir/some_dir to be
// created. This does not work; pass 'dest_dir/some_dir' instead.
symlinkOrCopySync('src_dir/some_dir', 'dest_dir');
```

It is an error if the parent directory of `destPath` does not already exist.

When we symlink, if the file at `srcPath` is a symlink as well, it will be
dereferenced before symlinking, to avoid runaway symlink indirection.

## Notes

* Symlinks could technically work on Windows, but they require special rights.
There are also junctions, but it's not clear whether they are useful. We might
want to be smarter about using symlinks on Windows when we can, but at the
moment we opt for the simplest solution (always copying), even though it
sacrifices performance on Windows.

* There intentionally isn't an asynchronous version. It's not clear that we
need or want one. Before sending a patch to add an async version, please share
your use case on the issue tracker.
