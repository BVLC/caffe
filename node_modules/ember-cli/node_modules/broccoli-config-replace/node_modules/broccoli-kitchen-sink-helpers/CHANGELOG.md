# master

# 0.3.1

* Fix Unicode handling for `hashStrings` function

# 0.3.0

* In keysForTree, require top-level directory (or file) to exist

# 0.2.9

* Update mkdirp

# 0.2.8

* Export `keysForTree`

# 0.2.7

* Update glob and use new `follow` option

# 0.2.6

* Lock down to `node-glob` 4.0.4. After 4.0.4 `node-glob` changed the symlinked directory
  behavior in a way that breaks a number of downstream broccoli plugins.

# 0.2.5

* Follow symlinks in `copyRecursivelySync`, `copyPreserveSync`, and `hashTree`
* Update `node-glob` to latest (4.0.5).

# 0.2.4

* Validate that `multiGlob` argument is an array

# 0.2.3

* Speed up `hashStrings` by using `MD5` (instead of `SHA256`).
* Add `symlinkOrCopyPreserveSync` for symlinking with copy fallback on Windows

# 0.2.2

* Make `hashTree` match after `copyRecursivelySync`, by disregarding the
  name of the root directory and not including directory times

# 0.2.1

* Make readdir ordering deterministic

# 0.2.0

* Remove `linkRecursivelySync` & `linkAndOverwrite`

# 0.1.2

* Add `copyRecursivelySync` & `copyPreserveSync`
* Change `linkRecursivelySync` & `linkAndOverwrite` to use
  `copyRecursivelySync` & `copyPreserveTime` respectively. We now refuse
  to overwrite in either of those functions, despite the `linkAndOverwrite`
  name.

# 0.1.1

* In `linkRecursivelySync`, link (broken) symlinks correctly

# 0.1.0

* Initial release
