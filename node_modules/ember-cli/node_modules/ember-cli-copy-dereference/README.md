# node-copy-dereference

Copy a file or directory, dereferencing symlinks in the process, and
preserving last-modified times and file modes.

Made for use by Broccoli and Broccoli plugins.

## Installation

```sh
npm install --save copy-dereference
```

## Example

```js
var copyDereferenceSync = require('copy-dereference').sync;

copyDereferenceSync('src_dir/some_file.txt', 'dest_dir/some_file.txt');
copyDereferenceSync('src_dir/some_dir', 'dest_dir/some_dir');
```

## Description

```js
copyDereferenceSync(srcPath, destPath)
```

Copy the file or directory at `srcPath` to `destPath`.

If `srcPath` is a symlink, or if there is a symlink somewhere underneath the
directory at `srcPath`, it will be dereferenced, that is, it will be replaced
with the thing it points to.

File & directory last-modified times as well as file modes (permissions &
executable bit) will be preserved.

We throw an exception if there are any broken symlinks at or beneath
`srcPath`, if `srcPath` does not exist, of if `destPath`'s parent directory
does not exist.

Furthermore, we throw an exception if `destPath` already exists. Thus in
contrast to Unix `cp`, the following will fail:

```js
// dest_dir already exists, and we might expect dest_dir/some_dir to be
// created. This does not work; pass 'dest_dir/some_dir' instead.
copyDereferenceSync('src_dir/some_dir', 'dest_dir');
```

File types other than files, directories and symlinks (such as device files or
sockets) are not supported and will cause an exception.

## Notes

* There intentionally isn't an asynchronous version. It's not clear that we
need or want one. Before sending a patch to add an async version, please share
your use case on the issue tracker.
