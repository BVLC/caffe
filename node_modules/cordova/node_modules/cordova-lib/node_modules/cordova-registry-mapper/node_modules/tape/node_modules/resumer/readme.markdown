# resumer

Return a through stream that starts out paused and resumes on the next tick,
unless somebody called `.pause()`.

This module has the same signature as
[through](https://npmjs.com/package/through).

[![browser support](https://ci.testling.com/substack/resumer.png)](http://ci.testling.com/substack/resumer)

[![build status](https://secure.travis-ci.org/substack/resumer.png)](http://travis-ci.org/substack/resumer)

# example

``` js
var resumer = require('resumer');
var s = createStream();
s.pipe(process.stdout);

function createStream () {
    var stream = resumer();
    stream.queue('beep boop\n');
    return stream;
}
```

```
$ node example/resume.js
beep boop
```

# methods

``` js
var resumer = require('resumer')
```

## resumer(write, end)

Return a new through stream from `write` and `end`, which default to
pass-through `.queue()` functions if not specified.

The stream starts out paused and will be resumed on the next tick unless you
call `.pause()` first.

`write` and `end` get passed directly through to
[through](https://npmjs.com/package/through).

# install

With [npm](https://npmjs.org) do:

```
npm install resumer
```

# license

MIT
