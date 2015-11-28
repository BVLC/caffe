Fast Source Map Concatenation
-----------------------------

This library lets you concatenate files (with or without their own
pre-generated sourcemaps), and get a single output file along with a
sourcemap.

It was written for use in ember-cli via broccoli-sourcemap-concat.

source-map dependency
---------------------

We depend on mozilla's source-map library, but only to use their
base64-vlq implementation, which is in turn based on the version in
the Closure Compiler. 

We can concatenate much faster than source-map because we are
specifically optimized for line-by-line concatenation.
