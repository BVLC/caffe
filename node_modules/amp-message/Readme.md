
# amp-message

  High level [AMP](https://github.com/visionmedia/node-amp) `Message` implementation for manipulating, encoding and decoding AMP messages.

## Installation

```
$ npm install amp-message
```

## Example

  Encoding a message:

```js
var Message = require('amp-message');

var msg = new Message;

console.log(msg.toBuffer());
// => <Buffer 10>

msg.push('foo');
msg.push('bar');
msg.push('baz');
console.log(msg.toBuffer());
// => <Buffer 13 00 05 73 3a 66 6f 6f 00 05 73 3a 62 61 72 00 05 73 3a 62 61 7a>

msg.push({ foo: 'bar' });
console.log(msg.toBuffer());
// => <Buffer 14 00 05 73 3a 66 6f 6f 00 05 73 3a 62 61 72 00 05 73 3a 62 61 7a 00 0f 6a 3a 7b 22 66 6f 6f 22 3a 22 62 61 72 22 7d>

msg.push(new Buffer('image data'));
console.log(msg.toBuffer());
// => <Buffer 15 00 05 73 3a 66 6f 6f 00 05 73 3a 62 61 72 00 05 73 3a 62 61 7a 00 0f 6a 3a 7b 22 66 6f 6f 22 3a 22 62 61 72 22 7d 00 0a 69 6d 61 67 65 20 64 61 74 ... >
```

  Decoding a message:

```js
var Message = require('..');

var msg = new Message;

msg.push('foo')
msg.push({ hello: 'world' })
msg.push(new Buffer('hello'))

var other = new Message(msg.toBuffer());

console.log(other.shift());
console.log(other.shift());
console.log(other.shift());
```

## API

### Message

  Initialize an empty message.

### Message(buffer)

  Decode the `buffer` AMP message to populate the `Message`.

### Message(args)

  Initialize a messeage populated with `args`.

# License

  MIT