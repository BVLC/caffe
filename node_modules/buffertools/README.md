# node-buffertools

Utilities for manipulating buffers.

## Installing the module

Easy! With [npm](http://npmjs.org/):

	npm install buffertools

From source:

	node-gyp configure
	node-gyp build

Now you can include the module in your project.

	require('buffertools').extend();  // extend Buffer.prototype
	var buf = new Buffer(42);         // create a 42 byte buffer
	buf.clear();                      // clear it!

If you don't want to extend the Buffer class's prototype (recommended):

	var buffertools = require('buffertools');
	var buf = new Buffer(42);
	buffertools.clear(buf);

## Methods

Note that most methods that take a buffer as an argument, will also accept a string.

### buffertools.extend([object], [object...])

Extend the arguments with the buffertools methods.  If called without arguments,
defaults to `[Buffer.prototype, SlowBuffer.prototype]`.  Extending prototypes
only makes sense for classes that derive from `Buffer`.

buffertools v1.x extended the `Buffer` prototype by default.  In v2.x, it is
opt-in.  The reason for that is that buffertools was originally developed for
node.js v0.3 (or maybe v0.2, I don't remember exactly when buffers were added)
where the `Buffer` class was devoid of any useful methods.  Over the years, it
has grown a number of utility methods, some of which conflict with the
buffertools methods of the same name, like `Buffer#fill()`.

### Buffer#clear()
### buffertools.clear(buffer)

Clear the buffer. This is equivalent to `Buffer#fill(0)`.
Returns the buffer object so you can chain method calls.

### Buffer#compare(buffer|string)
### buffertools.compare(buffer, buffer|string)

Lexicographically compare two buffers. Returns a number less than zero
if a < b, zero if a == b or greater than zero if a > b.

Buffers are considered equal when they are of the same length and contain
the same binary data.

Smaller buffers are considered to be less than larger ones. Some buffers
find this hurtful.

### Buffer#concat(a, b, c, ...)
### buffertools.concat(a, b, c, ...)

Concatenate two or more buffers/strings and return the result. Example:

	// identical to new Buffer('foobarbaz')
	a = new Buffer('foo');
	b = new Buffer('bar');
	c = a.concat(b, 'baz');
	console.log(a, b, c); // "foo bar foobarbaz"

	// static variant
	buffertools.concat('foo', new Buffer('bar'), 'baz');

### Buffer#equals(buffer|string)
### buffertools.equals(buffer, buffer|string)

Returns true if this buffer equals the argument, false otherwise.

Buffers are considered equal when they are of the same length and contain
the same binary data.

Caveat emptor: If your buffers contain strings with different character encodings,
they will most likely *not* be equal.

### Buffer#fill(integer|string|buffer)
### buffertools.fill(buffer, integer|string|buffer)

Fill the buffer (repeatedly if necessary) with the argument.
Returns the buffer object so you can chain method calls.

### Buffer#fromHex()
### buffertools.fromHex(buffer)

Assumes this buffer contains hexadecimal data (packed, no whitespace)
and decodes it into binary data. Returns a new buffer with the decoded
content. Throws an exception if non-hexadecimal data is encountered.

### Buffer#indexOf(buffer|string, [start=0])
### buffertools.indexOf(buffer, buffer|string, [start=0])

Search this buffer for the first occurrence of the argument, starting at
offset `start`. Returns the zero-based index or -1 if there is no match.

### Buffer#reverse()
### buffertools.reverse(buffer)

Reverse the content of the buffer in place. Example:

	b = new Buffer('live');
	b.reverse();
	console.log(b); // "evil"

### Buffer#toHex()
### buffertools.toHex(buffer)

Returns the contents of this buffer encoded as a hexadecimal string.

## Classes

Singular, actually. To wit:

## WritableBufferStream

This is a regular node.js [writable stream](http://nodejs.org/docs/v0.3.4/api/streams.html#writable_Stream)
that accumulates the data it receives into a buffer.

Example usage:

	// slurp stdin into a buffer
	process.stdin.resume();
	ostream = new WritableBufferStream();
	util.pump(process.stdin, ostream);
	console.log(ostream.getBuffer());

The stream never emits 'error' or 'drain' events.

### WritableBufferStream.getBuffer()

Return the data accumulated so far as a buffer.

## TODO

* Logical operations on buffers (AND, OR, XOR).
* Add lastIndexOf() functions.

## License

Copyright (c) 2010, Ben Noordhuis <info@bnoordhuis.nl>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
