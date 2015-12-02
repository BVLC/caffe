This is [Unicode Normalizer] in a Common JS module. I'm not affiliated with Matsuza, the original author of Unicode Normalizer.

[![Build Status](https://travis-ci.org/walling/unorm.png?branch=master)](https://travis-ci.org/walling/unorm)


Installation
------------

```bash
npm install unorm
```

Polyfill
--------

You can use this module as a polyfill for [String.prototype.normalize], for example:

```javascript
console.log('æøåäüö'.normalize('NFKD'));
```

The module uses some [EcmaScript 5](http://kangax.github.io/es5-compat-table/) features. Other browsers should use a compability shim, e.g. [es5-shim](https://github.com/kriskowal/es5-shim).

Functions
---------

This module exports four functions: `nfc`, `nfd`, `nfkc`, and `nfkd`; one for each Unicode normalization. In the browser the functions are exported in the `unorm` global. In CommonJS environments you just require the module. Functions:

 *  `unorm.nfd(str)` – Canonical Decomposition
 *  `unorm.nfc(str)` – Canonical Decomposition, followed by Canonical Composition
 *  `unorm.nfkd(str)` – Compatibility Decomposition
 *  `unorm.nfkc(str)` – Compatibility Decomposition, followed by Canonical Composition


Node.JS example
---------------

For a longer example, see `examples` directory.

```javascript
var unorm = require('unorm');

var text =
  'The \u212B symbol invented by A. J. \u00C5ngstr\u00F6m ' +
  '(1814, L\u00F6gd\u00F6, \u2013 1874) denotes the length ' +
  '10\u207B\u00B9\u2070 m.';

var combining = /[\u0300-\u036F]/g; // Use XRegExp('\\p{M}', 'g'); see example.js.

console.log('Regular:  ' + text);
console.log('NFC:      ' + unorm.nfc(text));
console.log('NFD:      ' + unorm.nfd(text));
console.log('NFKC:     ' + unorm.nfkc(text));
console.log('NFKD: *   ' + unorm.nfkd(text).replace(combining, ''));
console.log(' * = Combining characters removed from decomposed form.');
```


Road map
--------

As of November 2013. Longer term:

- Look at possible optimizations (speed primarely, module size secondarily)
- Adding functions to quick check normalizations: `is_nfc`, `is_nfd`, etc.


Contributers
------------

 - **Oleg Grenrus** is helping to maintain this library. He cleaned up the code base, fixed JSHint errors, created a test suite and updated the normalization data to Unicode 6.3.


Development notes
-----------------

- [Unicode normalization forms report](http://www.unicode.org/reports/tr15/)
- Unicode data can be found from http://www.unicode.org/Public/UCD/latest/ucd

To generate new unicode data, run:
```sh
cd src/data/src
javac UnormNormalizerBuilder.class
java UnormNormalizerBuilder
```
produced `unormdata.js` contains needed table

Execute `node benchmark/benchmark.js` to run simple benchmarks, if you do any changes which may affect performance.

License
-------

This project includes the software package **Unicode Normalizer 1.0.0**. The
software dual licensed under the MIT and GPL licenses. Here is the MIT license:

    Copyright (c) 2008-2013 Matsuza <matsuza@gmail.com>, Bjarke Walling <bwp@bwp.dk>
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.


[Unicode Normalizer]: http://coderepos.org/share/browser/lang/javascript/UnicodeNormalizer
[String.prototype.normalize]: http://people.mozilla.org/~jorendorff/es6-draft.html#sec-15.5.3.26
