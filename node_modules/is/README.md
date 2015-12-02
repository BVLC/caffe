# is

The definitive JavaScript type testing library

To be or not to be? This is the library!

[![browser support][1]][2]

## Installation

As a node.js module

    $ npm install is

As a component

    $ component install enricomarino/is

## API

### general

 - is.a (value, type) or is.type (value, type)
 - is.defined (value)
 - is.empty (value)
 - is.equal (value, other)
 - is.hosted (value, host)
 - is.instance (value, constructor)
 - is.instanceof (value, constructor) - deprecated, because in ES3 browsers, "instanceof" is a reserved word
 - is.null (value) - deprecated, because in ES3 browsers, "null" is a reserved word
 - is.undefined (value)

### arguments

 - is.arguments (value)
 - is.arguments.empty (value)

### array

 - is.array (value)
 - is.array.empty (value)
 - is.arraylike (value)

### boolean

 - is.boolean (value)
 - is.false (value) - deprecated, because in ES3 browsers, "false" is a reserved word
 - is.true (value) - deprecated, because in ES3 browsers, "true" is a reserved word

### date

 - is.date (value)

### element

 - is.element (value)

### error

 - is.error (value)

### function

 - is.fn(value)
 - is.function(value) - deprecated, because in ES3 browsers, "function" is a reserved word

### number

 - is.number (value)
 - is.infinite (value)
 - is.decimal (value)
 - is.divisibleBy (value, n)
 - is.int (value)
 - is.maximum (value, others)
 - is.minimum (value, others)
 - is.nan (value)
 - is.even (value)
 - is.odd (value)
 - is.ge (value, other)
 - is.gt (value, other)
 - is.le (value, other)
 - is.lt (value, other)
 - is.within (value, start, finish)

### object

 - is.object (value)

### regexp

 - is.regexp (value)

### string

 - is.string (value)


## Contributors

- [Jordan Harband](https://github.com/ljharb)

## License

(The MIT License)

Copyright (c) 2013 Enrico Marino

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

[1]: https://ci.testling.com/enricomarino/is.png
[2]: https://ci.testling.com/enricomarino/is

