Intro
=====

`base64-js` does basic base64 encoding/decoding in pure JS. Many browsers already have this functionality, but it is for text data, not all-purpose binary data.

Sometimes encoding/decoding binary data in the browser is useful, and that is what this module does.

API
===

`base64-js` has two exposed functions, `toByteArray` and `fromByteArray`, which both take a single argument.

* toByteArray- Takes a base64 string and returns a byte array
* fromByteArray- Takes a byte array and returns a base64 string
