CPR (cp -R)
===========

There are other modules out there that attempt this,
but none did it the way I needed it to be done or
they had issues and the author failed to fix them in a
timely manner.


Install
-------

    npm install cpr


Build Status
------------

[![Build Status](https://secure.travis-ci.org/davglass/cpr.png)](http://travis-ci.org/davglass/cpr)

Usage
-----

```js
var cpr = require('cpr');
//or
var cpr = require('cpr').cpr; //Back compat

cpr('/path/from', '/path/to', {
    deleteFirst: true, //Delete "to" before
    overwrite: true, //If the file exists, overwrite it
    confirm: true //After the copy, stat all the copied files to make sure they are there
}, function(err, files) {
    //err - The error if any (err.list might be available with an array of errors for more detailed information)
    //files - List of files that we copied
});

cpr('/path/from', '/path/to', function(err, files) {
    //err - The error if any (err.list might be available with an array of errors for more detailed information)
    //files - List of files that we copied
});
```

Options
-------

   * `deleteFirst`: Delete the to directory with `rimraf`
   * `overwrite`: If the destination exists, overwrite it
   * `confirm`: After the copy operation, stat all the files and report errors if any are missing
   * `filter`: `RegExp` or `function` to test each file against before copying


Filtering
---------

If you give it a `RegExp`, it will use that to `test` the filename as they are being gathered. If it passes, it will be removed.
If you give it a `function`, it will use that with `Array.filter` on the list of files.


![cpr](../master/cpr.jpg?raw=true)
