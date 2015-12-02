
# amp

  Abstract Message Protocol codec and streaming parser for nodejs.

## Installation

```
$ npm install amp
```

## Example

```js
var bin = amp.encode([new Buffer('hello'), new Buffer('world')]);
var msg = amp.decode(bin);
console.log(msg);
```

## Protocol

  AMP is a simple versioned protocol for framed messages containing
  zero or more "arguments". Each argument is opaque binary, thus you
  may use JSON, BSON, msgpack and others on top of AMP. Multiple argument
  support is used to allow a hybrid of binary/non-binary message args without
  requiring higher level serialization libraries like msgpack or BSON.

  All multi-byte integers are big endian. The `version` and `argc` integers
  are stored in the first byte, followed by a sequence of zero or more
  `<length>` / `<data>` pairs, where `length` is a 32-bit unsigned integer.

```
      0        1 2 3 4     <length>    ...
+------------+----------+------------+
| <ver/argc> | <length> | <data>     | additional arguments
+------------+----------+------------+
```

# License

  MIT