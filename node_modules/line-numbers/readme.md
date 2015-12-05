Overview [![Build Status](https://travis-ci.org/lydell/line-numbers.svg?branch=master)](https://travis-ci.org/lydell/line-numbers)
========

Add line numbers to a string.

```js
var lineNumbers = require("line-numbers")

var string = [
  "function sum(a, b) {",
  "  return a + b;",
  "}"
].join("\n")

lineNumbers(string)
// 1 | function sum(a, b) {
// 2 |   return a + b;
// 3 | }
```


Installation
============

- `npm install line-numbers`

```js
var lineNumbers = require("line-numbers")
```


Usage
=====

### `lineNumbers(code, [options])` ###

Inserts a line number at the beginning of each line in `code`, which is either a
string or an array of strings—one for each line. All the line numbers are of the
same width; shorter numbers are padded on the left side.

The return value is of the same type as `code`.

`options`:

- start: `Number`. The number to use for the first line. Defaults to `1`.
- padding: `String`. The character to pad numbers with. Defaults to `" "`.
- before: `String`. String to put before the line number. Defaults to `" "`.
- after: `String`. String to put between the line number and the line itself.
  Defaults to `" | "`.
- transform: `Function`. It is called for each line and passed an object with
  the following properties:

  - before: `options.before`
  - number: `Number`. The current line number.
  - width: `Number`. The padded width of the line numbers.
  - after: `options.after`
  - line: `String`. The current line.

  You may modify the above properties to alter the line numbering for the
  current line. This is useful if `before` and `after` aren’t enough, if you
  want to colorize the line numbers, or highlight the current line.


License
=======

[The X11 (“MIT”) License](LICENSE).
