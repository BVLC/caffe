# readable-wrap

upgrade streams1 to streams2 streams as a standalone module

This module provides a wrap function based on `Readable().wrap()` from node core
but as a standalone module.

Use this module if you don't want to wait for
[a patch in node core](https://github.com/joyent/node/pull/7758)
to land that fixes falsey objectMode values in wrapped readable streams.

[![build status](https://secure.travis-ci.org/substack/readable-wrap.png)](http://travis-ci.org/substack/readable-wrap)

[![testling badge](https://ci.testling.com/substack/readable-wrap.png)](https://ci.testling.com/substack/readable-wrap)

# example

``` js
var split = require('split');
var wrap = require('readable-wrap');
var through = require('through2');

process.stdin.pipe(wrap.obj(split())).pipe(through.obj(write));

function write (buf, enc, next) {
    console.log(buf.length + ': ' + buf);
    next();
}
```

output:

```
$ echo -e 'one\ntwo\n\nthree' | node example/split.js 
3: one
3: two
0: 
5: three
0: 
```

In object mode you get the empty lines, which is handy if you need to perform a
special action on empty lines such as to partition an HTTP request header from a
body in a streaming fashion.

In non-object mode the empty lines get ignored because that is how node core
streams work.

# methods

``` js
var wrap = require('readable-wrap')
```

## var stream = wrap(oldStream, opts)

Return a new streams2 `stream` based on the streams1 stream `oldStream`.

The `opts` will be passed to the underlying readable stream instance.

## var stream = wrap.obj(oldStream, opts)

Return a new streams2 `stream` based on the streams1 stream `oldStream` with
`opts.objectMode` set to `true`.

# install

With [npm](https://npmjs.org) do:

```
npm install readable-wrap
```

# license

MIT
