
# is-type

Type checking from node core.

This basically is [core-util-is](https://github.com/isaacs/core-util-is)
but with a nicer api.

## Example

```js
var is = require('is-type');

is.array([1]); // => true
is.primitive(true); // => true
is.primitive({}); // => false
```

## API

### is.array(arr)
### is.boolean(bool)
### is.null(null)
### is.nullOrUndefined(null)
### is.number(num)
### is.string(str)
### is.symbol(sym)
### is.undefined(undef)
### is.regExp(reg)
### is.object(obj)
### is.date(date)
### is.error(err)
### is.function(fn)
### is.primitive(prim)
### is.buffer(buf)

## Installation

With [npm](https://npmjs.org) do:

```bash
npm install is-type
```

## License

(MIT)

Copyright (c) 2013 Julian Gruber &lt;julian@juliangruber.com&gt;

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
