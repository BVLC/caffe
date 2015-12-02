# Synposis
An elegant way to define lightweight protocols on-top of TCP/TLS sockets in node.js 

# Motivation
Working within node.js it is very easy to write lightweight network protocols that communicate over TCP or TLS. The definition of such protocols often requires repeated (and tedious) parsing of individual TCP/TLS packets into a message header and some JSON body.

# Build Status
[![Build Status](https://secure.travis-ci.org/nodejitsu/nssocket.png)](http://travis-ci.org/nodejitsu/nssocket)

# Installation
```
  [sudo] npm install nssocket
```

# How it works

With `nssocket` this tedious bookkeeping work is done automatically for you in two ways:

1. Leverages wildcard and namespaced events from [EventEmitter2][0]
2. Automatically serializes messages passed to `.send()` and deserializes messages from `data` events.
3. Implements default reconnect logic for potentially faulty connections.
4. Automatically wraps TCP connections with TLS using [a known workaround][1]

## Messages
Messages in `nssocket` are serialized JSON arrays of the following form:

``` js
  ["namespace", "to", "event", { "this": "is", "the": "payload" }]
```

Although this is not as optimal as other message formats (pure binary, msgpack) most of your applications are probably IO-bound, and not by the computation time needed for serialization / deserialization. When working with `NsSocket` instances, all events are namespaced under `data` to avoid collision with other events.

## Simple Example
``` js
  var nssocket = require('nssocket');

  //
  // Create an `nssocket` TCP server
  //
  var server = nssocket.createServer(function (socket) {
    //
    // Here `socket` will be an instance of `nssocket.NsSocket`.
    //
    socket.send(['you', 'there']);
    socket.data(['iam', 'here'], function (data) {
      //
      // Good! The socket speaks our language 
      // (i.e. simple 'you::there', 'iam::here' protocol)
      //
      // { iam: true, indeedHere: true }
      //
      console.dir(data);
    })
  });
  
  //
  // Tell the server to listen on port `6785` and then connect to it
  // using another NsSocket instance.
  //
  server.listen(6785);
  
  var outbound = new nssocket.NsSocket();
  outbound.data(['you', 'there'], function () {
    outbound.send(['iam', 'here'], { iam: true, indeedHere: true });
  });
  
  outbound.connect(6785);
```

## Reconnect Example
`nssocket` exposes simple options for enabling reconnection of the underlying socket. By default, these options are disabled. Lets look at a simple example:

``` js
  var net = require('net'),
      nssocket = require('nssocket');
  
  net.createServer(function (socket) {
    //
    // Close the underlying socket after `1000ms`
    //
    setTimeout(function () {
      socket.destroy();
    }, 1000);
  }).listen(8345);
  
  //
  // Create an NsSocket instance with reconnect enabled
  //
  var socket = new nssocket.NsSocket({
    reconnect: true,
    type: 'tcp4',
  });
  
  socket.on('start', function () {
    //
    // The socket will emit this event periodically
    // as it attempts to reconnect
    //
    console.dir('start');
  });
  
  socket.connect(8345);
```

# API

### socket.send(event, data) 
Writes `data` to the socket with the specified `event`, on the receiving end it will look like: `JSON.stringify([event, data])`.

### socket.on(event, callback)
Equivalent to the underlying `.addListener()` or `.on()` function on the underlying socket except that it will permit all `EventEmitter2` wildcards and namespaces.

### socket.data(event, callback)
Helper function for performing shorthand listeners namespaced under the `data` event. For example:

``` js
  //
  // These two statements are equivalent
  //
  someSocket.on(['data', 'some', 'event'], function (data) { });
  someSocket.data(['some', 'event'], function (data) { });
```

### socket.end()
 Closes the current socket, emits `close` event, possibly also `error`

### socket.destroy()
 Remove all listeners, destroys socket, clears buffer. It is recommended that you use `socket.end()`.

## Tests
All tests are written with [vows][2] and should be run through [npm][3]:

``` bash
  $ npm test
```

### Author: [Nodejitsu](http://www.nodejitsu.com)
### Contributors: [Paolo Fragomeni](http://github.com/hij1nx), [Charlie Robbins](http://github.com/indexzero), [Jameson Lee](http://github.com/drjackal), [Gene Diaz Jr.](http://github.com/genediazjr)
 
[0]: http://github.com/hij1nx/eventemitter2
[1]: https://gist.github.com/848444
[2]: http://vowsjs.org
[3]: http://npmjs.org
