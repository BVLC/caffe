JSONSelect is *EXPERIMENTAL*, *ALPHA*, etc.

JSONSelect defines a selector language similar to CSS intended for
JSON documents.  For an introduction to the project see
[jsonselect.org](http://jsonselect.org) or the [documentation](https://github.com/lloyd/JSONSelect/blob/master/JSONSelect.md).

## Project Overview

JSONSelect is an attempt to create a selector language similar to
CSS for JSON objects.  A couple key goals of the project's include:

  * **intuitive** - JSONSelect is meant to *feel like* CSS, meaning a developers with an understanding of CSS can probably guess most of the syntax.
  * **expressive** - As JSONSelect evolves, it will include more of the most popular constructs from the CSS spec and popular implementations (like [sizzle](http://sizzlejs.com/)).  A successful result will be a good balance of simplicity and power.
  * **language independence** - The project will avoid features which are unnecessarily tied to a particular implementation language.
  * **incremental adoption** - JSONSelect features are broken in to conformance levels, to make it easier to build basic support and to allow incremental stabilization of the language.
  * **efficient** - As many constructs of the language as possible will be able to be evaluated in a single document traversal.  This allows for efficient stream filtering.

JSONSelect should make common operations easy, complex operations possible,
but haughtily ignore weird shit.

## What's Here

This repository is the home to many things related to JSONSelect:

  * [Documentation](https://github.com/lloyd/JSONSelect/blob/master/JSONSelect.md) which describes the language
  * The [jsonselect.org](http://jsonselect.org) [site source](https://github.com/lloyd/JSONSelect/blob/master/site/)
  * A [reference implementation](https://github.com/lloyd/JSONSelect/blob/master/src/jsonselect.js) in JavaScript

## Related projects

Conformance tests are broken out into a [separate
repository](https://github.com/lloyd/JSONSelectTests) and may be used
by other implementations.