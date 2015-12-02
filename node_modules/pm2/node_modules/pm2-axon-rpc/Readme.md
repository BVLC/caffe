
# axon-rpc

[![Build Status](https://travis-ci.org/unitech/pm2-axon-rpc.png)](https://travis-ci.org/unitech/pm2-axon-rpc)

  RPC client / server for [axon](https://github.com/visionmedia/axon).

## arpc(1)

  The `arpc(1)` executable allows you to expose entire
  node modules with a single command, or inspect
  methods exposed by a given node.

```

Usage: arpc [options] <module>

Options:

  -h, --help            output usage information
  -V, --version         output the version number
  -a, --addr <addr>     bind to the given <addr>
  -m, --methods <addr>  inspect methods exposed by <addr>

```

## Server

```js
var rpc = require('axon-rpc')
  , axon = require('axon')
  , rep = axon.socket('rep');

var server = new rpc.Server(rep);
rep.bind(4000);
```

### Server#expose(name, fn)

  Expose a single method `name` mapped to `fn` callback.

```js
server.expose('add', function(a, b, fn){
  fn(null, a + b);
});
```

### Server#expose(object)

  Expose several methods:

```js
server.expose({
  add: function(){ ... },
  sub: function(){ ... }
});
```

  This may also be used to expose
  an entire node module with exports:

```js
server.expose(require('./api'));
```

## Client

```js
var rpc = require('axon-rpc')
  , axon = require('axon')
  , req = axon.socket('req');

var client = new rpc.Client(req);
req.connect(4000);
```

### Client#call(name, ..., fn)

  Invoke method `name` with some arguments and invoke `fn(err, ...)`:

```js
client.call('add', 1, 2, function(err, n){
  console.log(n);
  // => 3
})
```

### Client#methods(fn)

  Request available methods:

```js
client.methods(function(err, methods){
  console.log(methods);
})
```

  Responds with objects such as:

```js
{
  add: {
    name: 'add',
    params: ['a', 'b', 'fn']
  }
}
```

## License

(The MIT License)

Copyright (c) 2014 TJ Holowaychuk &lt;tj@learnboost.com&gt;

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
