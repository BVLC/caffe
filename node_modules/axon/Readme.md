# Axon

  Axon is a message-oriented socket library for node.js heavily inspired by zeromq. For a light-weight
  UDP alternative you may be interested in [punt](https://github.com/tj/punt).

[![Build Status](https://travis-ci.org/tj/axon.png)](https://travis-ci.org/tj/axon)

## Installation

    $ npm install axon

## Features

  - message oriented
  - automated reconnection
  - light-weight wire protocol
  - mixed-type arguments (strings, objects, buffers, etc)
  - unix domain socket support
  - fast (~800 mb/s ~500,000 messages/s)

## Events

  - `close` when server or connection is closed
  - `error` (err) when an un-handled socket error occurs
  - `ignored error` (err) when an axon-handled socket error occurs, but is ignored
  - `socket error` (err) emitted regardless of handling, for logging purposes
  - `reconnect attempt` when a reconnection attempt is made
  - `connect` when connected to the peer, or a peer connection is accepted
  - `disconnect` when an accepted peer disconnects
  - `bind` when the server is bound
  - `drop` (msg) when a message is dropped due to the HWM
  - `flush` (msgs) queued when messages are flushed on connection

## Patterns

  - push / pull
  - pub / sub
  - req / rep
  - pub-emitter / sub-emitter

## Mixed argument types

  Backed by [node-amp-message](https://github.com/tj/node-amp-message)
  you may pass strings, objects, and buffers as arguments.

```js
push.send('image', { w: 100, h: 200 }, imageBuffer);
pull.on('message', function(type, size, img){});
```

## Push / Pull

`PushSocket`s distribute messages round-robin:

```js
var axon = require('axon');
var sock = axon.socket('push');

sock.bind(3000);
console.log('push server started');

setInterval(function(){
  sock.send('hello');
}, 150);
```

Receiver of `PushSocket` messages:

```js
var axon = require('axon');
var sock = axon.socket('pull');

sock.connect(3000);

sock.on('message', function(msg){
  console.log(msg.toString());
});
```


Both `PushSocket`s and `PullSocket`s may `.bind()` or `.connect()`. In the
following configuration the push socket is bound and pull "workers" connect
to it to receive work:

![push bind](http://f.cl.ly/items/473u3m1a0k1i0J0I3s04/ss-push.png)

This configuration shows the inverse, where workers connect to a "sink"
to push results:

![pull bind](http://f.cl.ly/items/3Y0j2v153Q0l1r373i0H/ss-pull.png)

## Pub / Sub

`PubSocket`s send messages to all subscribers without queueing. This is an
important difference when compared to a `PushSocket`, where the delivery of
messages will be queued during disconnects and sent again upon the next connection.

```js
var axon = require('axon');
var sock = axon.socket('pub');

sock.bind(3000);
console.log('pub server started');

setInterval(function(){
  sock.send('hello');
}, 500);
```

`SubSocket` simply receives any messages from a `PubSocket`:

```js
var axon = require('axon');
var sock = axon.socket('sub');

sock.connect(3000);

sock.on('message', function(msg){
  console.log(msg.toString());
});
```

 `SubSocket`s may optionally `.subscribe()` to one or more "topics" (the first multipart value),
 using string patterns or regular expressions:

```js
var axon = require('axon');
var sock = axon.socket('sub');

sock.connect(3000);
sock.subscribe('user:login');
sock.subscribe('upload:*:progress');

sock.on('message', function(topic, msg){

});
```

## Req / Rep

`ReqSocket` is similar to a `PushSocket` in that it round-robins messages
to connected `RepSocket`s, however it differs in that this communication is
bi-directional, every `req.send()` _must_ provide a callback which is invoked
when the `RepSocket` replies.

```js
var axon = require('axon');
var sock = axon.socket('req');

sock.bind(3000);

sock.send(img, function(res){

});
```

`RepSocket`s receive a `reply` callback that is used to respond to the request,
you may have several of these nodes.

```js
var axon = require('axon');
var sock = axon.socket('rep');

sock.connect(3000);

sock.on('message', function(img, reply){
  // resize the image
  reply(img);
});
```

 Like other sockets you may provide multiple arguments or an array of arguments,
 followed by the callbacks. For example here we provide a task name of "resize"
 to facilitate multiple tasks over a single socket:

```js
var axon = require('axon');
var sock = axon.socket('req');

sock.bind(3000);

sock.send('resize', img, function(res){

});
```

 Respond to the "resize" task:

```js
var axon = require('axon');
var sock = axon.socket('rep');

sock.connect(3000);

sock.on('message', function(task, img, reply){
  switch (task) {
    case 'resize':
      // resize the image
      reply(img);
      break;
  }
});
```

## PubEmitter / SubEmitter

  `PubEmitter` and `SubEmitter` are higher-level `Pub` / `Sub` sockets, using the "json" codec to behave much like node's `EventEmitter`. When a `SubEmitter`'s `.on()` method is invoked, the event name is `.subscribe()`d for you. Each wildcard (`*`) or regexp capture group is passed to the callback along with regular message arguments.

app.js:

```js
var axon = require('axon');
var sock = axon.socket('pub-emitter');

sock.connect(3000);

setInterval(function(){
  sock.emit('login', { name: 'tobi' });
}, 500);
```

logger.js:

```js
var axon = require('axon');
var sock = axon.socket('sub-emitter');

sock.bind(3000);

sock.on('user:login', function(user){
  console.log('%s signed in', user.name);
});

sock.on('user:*', function(action, user){
  console.log('%s %s', user.name, action);
});

sock.on('*', function(event){
  console.log(arguments);
});
```

## Socket Options

Every socket has associated options that can be configured via `get/set`.

  - `identity` - the "name" of the socket that uniqued identifies it.
  - `retry timeout` - connection retry timeout in milliseconds [100]
  - `retry max timeout` - the cap for retry timeout length in milliseconds [5000]
  - `hwm` - the high water mark threshold for queues [Infinity]

## Binding / Connecting

In addition to passing a portno, binding to INADDR_ANY by default, you
may also specify the hostname via `.bind(port, host)`, another alternative
is to specify the url much like zmq via `tcp://<hostname>:<portno>`, thus
the following are equivalent:

```
sock.bind(3000)
sock.bind(3000, '0.0.0.0')
sock.bind('tcp://0.0.0.0:3000')

sock.connect(3000)
sock.connect(3000, '0.0.0.0')
sock.connect('tcp://0.0.0.0:3000')
```

  You may also use unix domain sockets:

```
sock.bind('unix:///some/path')
sock.connect('unix:///some/path')
```

## Protocol

  Axon 2.x uses the extremely simple [AMP](https://github.com/tj/node-amp) protocol to send messages on the wire. Codecs are no longer required as they were in Axon 1.x.

## Performance

Preliminary benchmarks on my Macbook Pro based on 10 messages
per tick as a realistic production application would likely have
even less than this. "better" numbers may be acheived with batching
and a larger messages/tick count however this is not realistic.

  64 byte messages:

```

      min: 47,169 ops/s
     mean: 465,127 ops/s
   median: 500,000 ops/s
    total: 2,325,636 ops in 5s
  through: 28.39 mb/s

```

  1k messages:

```

      min: 48,076 ops/s
     mean: 120,253 ops/s
   median: 121,951 ops/s
    total: 601,386 ops in 5.001s
  through: 117.43 mb/s

```

  8k messages:

```

      min: 36,496 ops/s
     mean: 53,194 ops/s
   median: 50,505 ops/s
    total: 266,506 ops in 5.01s
  through: 405.84 mb/s

````

  32k messages:

```

      min: 12,077 ops/s
     mean: 14,792 ops/s
   median: 16,233 ops/s
    total: 74,186 ops in 5.015s
  through: 462.28 mb/s

```

## What's it good for?

  Axon are not meant to combat zeromq nor provide feature parity,
  but provide a nice solution when you don't need the insane
  nanosecond latency or language interoperability that zeromq provides
  as axon do not rely on any third-party compiled libraries.

## Running tests

```
$ npm install
$ make test
```

## Authors

  - [tj](http://github.com/tj)
  - [gjohnson](https://github.com/gjohnson)

## Links

  - [Screencast](https://vimeo.com/45818408)
  - [Axon RPC](https://github.com/tj/axon-rpc)

## License

  MIT
