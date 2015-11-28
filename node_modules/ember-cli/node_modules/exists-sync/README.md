exists-sync
===========
[![Build Status](https://travis-ci.org/ember-cli/exists-sync.svg)](https://travis-ci.org/ember-cli/exists-sync) [![Build status](https://ci.appveyor.com/api/projects/status/c05xyb4s80pn66yo?svg=true)](https://ci.appveyor.com/project/embercli/exists-sync)

Replacement for [fs.existsSync()](https://nodejs.org/api/fs.html#fs_fs_exists_path_callback), which is being deprecated: 

> "fs.exists() is an anachronism and exists only for historical reasons. There should almost never be a reason to use it in your own code.

> In particular, checking if a file exists before opening it is an anti-pattern that leaves you vulnerable to race conditions: another process may remove the file between the calls to fs.exists() and fs.open(). Just open the file and handle the error when it's not there."

`exists-sync` will recursively follow symlinks to verify the target file exists, rather than giving a false positive on a symlink whose target has been removed.