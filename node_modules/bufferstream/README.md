# BufferStream

painless stream buffering, cutting and piping.

## install

    npm install bufferstream

## api

BufferStream is a full node.js [Stream](http://nodejs.org/docs/v0.4.7/api/streams.html) so it has apis of both [Writeable Stream](http://nodejs.org/docs/v0.4.7/api/streams.html#writable_Stream) and [Readable Stream](http://nodejs.org/docs/v0.4.7/api/streams.html#readable_Stream).

### BufferStream

```javascript
BufferStream = require('bufferstream')
stream = new BufferStream([{encoding:'utf8', size:'none'}]) // default
```
 * `encoding` default encoding for writing strings
 * `blocking` if true and the source is a child_process the stream will block the entire process (timeouts wont work anymore, but splitting and listening on data still works, because they work sync)
 * `size` defines buffer level or sets buffer to given size (see ↓`setSize` for more)
 * `disabled` immediately call disable
 * `split` short form for:

```javascript
stream.split(token, function (chunk) {stream.emit('data', chunk)})
```

### stream.setSize

```javascript
stream.setSize(size) // can be one of ['none', 'flexible', <number>]
```

different buffer behaviors can be triggered by size:

 * `none` when output drains, bufferstream drains too
 * `flexible` buffers everthing that it gets and not piping out
 * `<number>` `TODO` buffer has given size. buffers everthing until buffer is full. when buffer is full then  the stream will drain

### stream.enable

```javascript
stream.enable()
```

enables stream buffering __default__

### stream.disable

```javascript
stream.disable()
```

flushes buffer and disables stream buffering.
BufferStream now pipes all data as long as the output accepting data.
when the output is draining BufferStream will buffer all input temporary.

```javascript
stream.disable(token, ...)
stream.disable(tokens) // Array
```
 * `token[s]` buffer splitters (should be String or Buffer)

disables given tokens. wont flush until no splitter tokens are left.

### stream.split

```javascript
stream.split(token, ...)
stream.split(tokens) // Array
```
 * `token[s]` buffer splitters (should be String or Buffer)

each time BufferStream finds a splitter token in the input data it will emit a __split__ event.
this also works for binary data.

### Event: 'split'

```javascript
stream.on('split', function (chunk, token) {…})
stream.split(token, function (chunk, token) {…}) // only get called for this particular token
```

whenever the stream is enabled it will try to find all splitter token in `stream.buffer`,
cut it off and emit the chunk (without token) as __split__ event.
this data will be lost when not handled.

the chunk is the cut off of `stream.buffer` without the token.

__Warning:__ try to avoid calling `stream.emit('data', newchunk)` more than one time, because this will likely throw `Error: Offset is out of bounds`.

### stream.getBuffer

```javascript
stream.getBuffer()
// or just
stream.buffer
```

returns its [Buffer](http://nodejs.org/docs/v0.4.7/api/buffers.html).

### stream.toString

```javascript
stream.toString()
```

shortcut for `stream.buffer.toString()`

### stream.length

```javascript
stream.length
```

shortcut for `stream.buffer.length`

### PostBuffer

```javascript
PostBuffer = require('bufferstream/postbuffer')
post = new PostBuffer(req)
```
 * `req` http.ServerRequest

for if you want to get all the post data from a http server request and do some db reqeust before.

buffer http client

### post.onEnd

```javascript
post.onEnd(function (data) {…});
```

set a callback to get all post data from a http server request

### post.pipe

```javascript
post.pipe(stream, options);
```

pumps data into another stream to allow incoming streams
given options will be passed to Stream.pipe

## note

To improve platform independence bufferstream is using `bufferjs` instead of `buffertools` since version `0.6.0`.
Just run `npm install buffertools` to use their implementation of `Buffer.indexOf` which is sligthly faster than `bufferjs`'s version.
if you're forced to use the javascript-only version of `Buffer.indexOf` (like on windows) you can disable the warning by:
```javascript
require('bufferstream').fn.warn = false
```

## example

```javascript
BufferStream = require('bufferstream')
stream = new BufferStream({encoding:'utf8', size:'flexible'})
stream.split('//', ':')
stream.on('split', function (chunk, token) {
    console.log("got '%s' by '%s'", chunk.toString(), token.toString())
})
stream.write("buffer:stream//23")
console.log(stream.toString())
```

results in

    got 'buffer' by ':'
    got 'stream' by '//'
    23

* https://github.com/dodo/node-bufferstream/blob/master/example/split.js

## FAQ

> I'm not sure from your readme what the split event emits?

you can specify more than one split token .. so it's emitted whenever
a token is found.

> does it emit the buffer up to the just before the token starts?

yes.

> also, does it join buffers together if they do not already end in a token?

when size is `flexible` it joins everything together what it gets to
one buffer (accessible through `stream.buffer` or
`stream.getBuffer()`)
whenever it gets data, it will try to find all tokens

> in other words, can I use this to rechunk a stream so that the chunks always break on newlines, for example?

yes.

```javascript
stream = new BufferStream({size:'flexible'});
stream.split('\n', function (line) { // line doesn't have a '\n' anymore
    stream.emit('data', line); // Buffer.isBuffer(line) === true
});
```

[![Build Status](https://secure.travis-ci.org/dodo/node-bufferstream.png)](http://travis-ci.org/dodo/node-bufferstream)
