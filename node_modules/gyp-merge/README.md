gyp-merge
=========

This module implements the data structure merging algortithms used in
evaluating includes and conditions in gyp files.

Usage
-----

    var gypMerge = require('gyp-merge')({noSingletons: false});

    console.log(gypMerge.mergeDictionary(
        {a: 1, b: 2, c: [3, 4, 5]},
        {a: 6, b: 7, "d+": [8, 9, 10]}
    ));

The extra `()` is configuration. Right now, the only option is `{ noSingletons:
true }`, which disables the "singleton" behavior when merging lists, keeping
order but eliding duplicates.
