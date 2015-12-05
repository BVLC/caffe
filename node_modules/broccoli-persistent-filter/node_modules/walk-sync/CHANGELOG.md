# master

# 0.2.6

* On Windows, normalize backslashes in root path to forward slashes

# 0.2.5

* Exclude all non-essential files from npm

# 0.2.4

* Fix file entries to have a numeric timestamp rather than a `Date`

# 0.2.3

* Extract matcher-collection into separate package

# 0.2.2

* Add `walkSync.entries`, which returns objects instead of files

# 0.2.1

* Add `directories` flag
* Allow passing the globs array as a `globs` option

# 0.2.0

* Add optional `globArray` parameter

# 0.1.3

* Switch to `fs.statSync` (instead of `fs.lstatSync`) to follow symlinks.

# 0.1.2

* Sort readdir entries for deterministic behavior

# 0.1.1

* Do not follow symlinks (as advertised)

# 0.1.0

* Bump version without change, to allow for caret/tilde dependencies

# 0.0.1

* Initial release
