gyp
===

This package is incomplete, but the goal of this package is first to become a
complete GYP reader, then perhaps implement some generators.

Usage
-----

    var gyp = require('gyp');
    gyp('test.gyp', {}, function(err, data) {
        // data is a plain object with gyp pre-phase variables expanded
        // and include files processed
    });

Still to do
-----------

* Handle the non-JSON parsing parts like comments
* Add a simple hook for post-phase variable expansions and handle build-specific use cases

Blue Sky
--------

I'd love to use gyp files in a devops platform as the recipe file format. It's
the right combination of structured and restrictive, and the include-loading
semantics are about right to give good modularity.
