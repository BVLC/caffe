# Broccoli: Potential Data Loss On OS X

*Posted April 7, 2014 by [Jo Liss](https://twitter.com/jo_liss)*

**Update May 4, 2014: All affected package versions have been removed from
npm. The original document is preserved below for posterity.**

There is an issue in Broccoli and several helper libraries that can cause data
loss on Mac OS X, stemming from their use of hardlinks.

## Background

Broccoli, as well as several plugins and helpers (plugin base classes) use
[hardlinks](https://en.wikipedia.org/wiki/Hard_link) as a fast way to copy
files unchanged from an input tree to an output tree, or copy them to and from
a cache directory.

It turns out that on OS X, it is possible to hardlink directories. While we
never intentionally try to do this, it is possible to accidentally hardlink
directories -- for instance when we hardlink a symlink pointing to a
directory.

Hardlinked directories can lead to data loss: If one of the directories
underneath the `tmp` directory is hardlinked to a directory outside the `tmp`
directory (say, containing source code), running `rm -r tmp` will delete all
the files in the outside directory as a side effect.

## Impact

Several libraries in Broccoli core, as well as plugins and helper base classes
used by plugins, use hardlinks. New versions that copy files instead of
hardlinking them have been released. The affected versions are listed below.

The data loss issue occurs on OS X, does not occur on Linux, and I don't know
whether it occurs on Windows.

## Versions Affected

* broccoli <= 0.7.0
* broccoli-kitchen-sink-helpers <= 0.1.1
* broccoli-filter <= 0.1.5
* broccoli-static-compiler <= 0.1.3
* broccoli-merge-trees <= 0.1.2
* broccoli-es6-import-validate <= 0.0.1

## Upgrading

You are advised to upgrade to package versions greater than those listed
above.

Even if your `package.json` does not depend on any of the affected package
versions directly, you might still be using them indirectly through some
Broccoli plugin. To check whether you are affected, run

```bash
npm install -g npm-check-affected
cd your-project-dir
npm install
npm-check-affected https://raw.githubusercontent.com/joliss/broccoli/master/docs/hardlink-issue.json
```

If you are using a plugin that depends on an older, affected version of some
package, please kindly send them a pull request updating their dependency.

## Possible `npm unpublish` In The Future

It is possible to encounter the data loss issue in the future, for instance if
you are using a plugin whose dependency spec hasn't been updated. I don't want
people to still have to worry about this problem months from now.

So I'm considering yanking all affected versions from npm using `npm
unpublish` at some point, perhaps two to three weeks from now.

The older versions would still be tagged on their git repositories if you
desperately need them. Note that npm (wisely) does not allow us to overwrite
previously released versions with fixed versions; we can only delete them.

Please let me know on [Twitter](https://twitter.com/jo_liss) or
[email](mailto:joliss42@gmail.com) whether or not you think this is a good
idea.

## Credits

Thanks to [@thomasABoyt](https://twitter.com/thomasABoyt) for reporting this
issue and helping me identify the cause.

*Questions? Comments? [Discuss on
Twitter](https://twitter.com/jo_liss/status/453240313583124480) or open a
GitHub issue.*
