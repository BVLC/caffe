# safe-clone-deep module for JavaScript

[![NPM version](https://badge.fury.io/js/safe-clone-deep.png)](
 https://www.npmjs.org/package/safe-clone-deep
) [![Build Status](https://travis-ci.org/tracker1/safe-clone-deep.png)](
 https://travis-ci.org/tracker1/safe-clone-deep
)

This module exposes a single function that accepts an object and clones it without circular references.

## Installation

```
npm install safe-clone-deep
```

For Browser and AMD support see the `dist/` directory.



## Usage

```javascript
// browser
// var clone = Object.safeCloneDeep

// node.js
var clone = require('safe-clone-deep');

var a = {};
a.a = a;
a.b = {};
a.b.a = a;
a.b.b = a.b;
a.c = {};
a.c.b = a.b;
a.c.c = a.c;
a.x = 1;
a.b.x = 2;
a.c.x = 3;
a.d = [0,a,1,a.b,2,a.c,3];

console.log(util.inspect(clone(a), {showHidden:false,depth:4}))
clone(a)
```

result...

```
{ a: undefined,
  b: { a: undefined, b: undefined, x: 2 },
  c: { b: { a: undefined, b: undefined, x: 2 }, c: undefined, x: 3 },
  x: 1,
  d:
   [ 0,
     undefined,
     1,
     { a: undefined, b: undefined, x: 2 },
     2,
     { b: { a: undefined, b: undefined, x: 2 }, c: undefined, x: 3 },
     3 ] }
```

### Override circularValue

With the `undefined` as the default circularValue, JSON.stringify will not keep the keys, which is likely the desired result.

```
JSON.stringify(clone(a));
```

result

```
{"b":{"x":2},"c":{"b":{"x":2},"x":3},"x":1,"d":[0,null,1,{"x":2},2,{"b":{"x":2},"x":3},3]}
```

NOTE: when the circular value is in an array, `null` is used instead of `undefined`.

--

You can override the default behavior.


```javascript
clone(a,'[Circular]');
```

result...

```
{ a: '[Circular]',
  b: { a: '[Circular]', b: '[Circular]', x: 2 },
  c:
   { b: { a: '[Circular]', b: '[Circular]', x: 2 },
     c: '[Circular]',
     x: 3 },
  x: 1,
  d:
   [ 0,
     '[Circular]',
     1,
     { a: '[Circular]', b: '[Circular]', x: 2 },
     2,
     { b: { a: '[Circular]', b: '[Circular]', x: 2 },
       c: '[Circular]',
       x: 3 },
     3 ] }
```

## License

The Internet Software Consortium License (ISC)

Copyright (c) 2014, Michael J. Ryan <tracker1@gmail.com>

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

