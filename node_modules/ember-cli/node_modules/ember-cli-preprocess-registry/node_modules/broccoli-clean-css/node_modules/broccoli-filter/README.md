# broccoli-filter

Helper base class for Broccoli plugins that map input files into output files
one-to-one.

Documentation to follow.

**Can this help with compilers that are almost 1:1, like a minifier that takes
a `.js` and `.js.map` file and outputs a `.js` and `.js.map` file?**

Not at the moment. I don't know yet how to implement this and still have the
API look beautiful. We also have to make sure that caching works correctly, as
we have to invalidate if either the `.js` or the `.js.map` file changes. My
plan is to write a source-map-aware uglifier plugin to understand this use
case better, and then extract common code back into this `Filter` base class.
