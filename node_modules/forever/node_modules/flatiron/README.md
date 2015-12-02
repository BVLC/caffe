# [flatiron](http://flatironjs.org) [![Build Status](https://secure.travis-ci.org/flatiron/flatiron.png)](http://travis-ci.org/flatiron/flatiron)

*Framework components for node.js and the browser*

![](http://flatironjs.org/img/flatiron.png)

# Example HTTP Server:

```js
var flatiron = require('flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.http);

app.router.get('/', function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.end('Hello world!\n');
});

app.start(8080);
```

# Example HTTPS Server:

```js
var flatiron = require('flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.http, {
  https: {
    cert: 'path/to/cert.pem',
    key: 'path/to/key.pem',
    ca: 'path/to/ca.pem'
  }
});

app.router.get('/', function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.end('Hello world!\n');
});

app.start(8080);
```

# Example CLI Application:

```js
// example.js

var flatiron = require('flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli, {
  dir: __dirname,
  usage: [
    'This is a basic flatiron cli application example!',
    '',
    'hello - say hello to somebody.'
  ]
});

app.cmd('hello', function () {
  app.prompt.get('name', function (err, result) {
    app.log.info('hello '+result.name+'!');
  })
})

app.start();
```

## Run It:

```
% node example.js hello
prompt: name: world
info:   hello world!
```

## Installation

### Installing NPM (Node Package Manager)
```
  curl http://npmjs.org/install.sh | sh
```

### Installing Flatiron
```
  [sudo] npm install flatiron
```

### Installing Union (Required for `flatiron.plugins.http`)
```
  npm install union
```

# Usage:

## Start With `flatiron.app`:

`flatiron.app` is a [broadway injection container](https://github.com/flatiron/broadway). To be brief, what it does is allow plugins to modify the `app` object directly:

```js
var flatiron = require('flatiron'),
    app = flatiron.app;

var hello = {
  attach: function (options) {
    this.hello = options.message || 'Why hello!';
  }
};

app.use(hello, {
  message: "Hi! How are you?"
});

// Will print, "Hi! How are you?"
console.log(app.hello);
```

Virtually all additional functionality in flatiron comes from broadway plugins, such as `flatiron.plugins.http` and `flatiron.plugins.cli`.

### `app.config`

`flatiron.app` comes with a [`config`](https://github.com/flatiron/broadway/blob/master/lib/broadway/plugins/config.js) plugin pre-loaded, which adds configuration management courtesy [nconf](https://github.com/flatiron/nconf). `app.config` has the same api as the `nconf` object.

The `literal` store is configured by default. If you want to use different stores you can easily attach them to the `app.config` instance.

```js
// add the `env` store to the config
app.config.use('env');

// add the `file` store the the config
app.config.use('file', { file: 'path/to/config.json' });

// or using an alternate syntax
app.config.env().file({ file: 'path/to/config.json' });

// and removing stores
app.config.remove('literal');
```

### `app.log`

`flatiron.app` will also load a [`log`](https://github.com/flatiron/broadway/blob/master/lib/broadway/plugins/log.js) plugin during the init phase, which attaches a [winston container](https://github.com/flatiron/winston) to `app.log`. This logger is configured by combining the `app.options.log` property with the configuration retrieved from `app.config.get('log')`.

## Create An HTTP Server with `flatiron.plugins.http(options)`:

This plugin adds http serving functionality to your flatiron app by attaching the following properties and methods:

### Define Routes with `app.router`:

This is a [director](https://github.com/flatiron/director) router configured to route http requests after the middlewares in `app.http.before` are applied. Example routes include:

```js

// GET /
app.router.get('/', function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.end('Hello world!\n');
});

// POST to /
app.router.post('/', function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' });
  this.res.write('Hey, you posted some cool data!\n');
  this.res.end(util.inspect(this.req.body, true, 2, true) + '\n');
});

// Parameterized routes
app.router.get('/sandwich/:type', function (type) {
  if (~['bacon', 'burger'].indexOf(type)) {
    this.res.writeHead(200, { 'Content-Type': 'text/plain' });
    this.res.end('Serving ' + type + ' sandwich!\n');
  }
  else {
    this.res.writeHead(404, { 'Content-Type': 'text/plain' });
    this.res.end('No such sandwich, sorry!\n');
  }
});
```

`app.router` can also route against regular expressions and more! To learn more about director's advanced functionality, visit director's [project page](https://github.com/flatiron/director#readme).


### Access The Server with `app.server`:

This is a [union](https://github.com/flatiron/union) middleware kernel.

### Modify the Server Options with `app.http`:

This object contains options that are passed to the union server, including `app.http.before`, `app.http.after` and `app.http.headers`.

These properties may be set by passing them through as options:

```js
app.use(flatiron.plugins.http, {
  before: [],
  after: []
});
```

You can read more about these options on the [union project page](https://github.com/flatiron/union#readme).

### Start The Server with `app.start(port, <host>, <callback(err)>)`

This method will both call `app.init` (which will call any asynchronous initialization steps on loaded plugins) and start the http server with the given arguments. For example, the following will start your flatiron http server on port 8080:

```js
app.start(8080);
```

## Create a CLI Application with `flatiron.plugins.cli(options)`

This plugin turns your app into a cli application framework. For example, [jitsu]
(https://github.com/nodejitsu/jitsu) uses flatiron and the cli plugin.

Valid options include:

```js
{
  "argvOptions": {}, // A configuration hash passed to the cli argv parser.
  "usage": [ "foo", "bar" ], // A message to show for cli usage. Joins arrays with `\n`.
  "dir": require('path').join(__dirname, 'lib', 'commands'), // A directory with commands to lazy-load
  "notFoundUsage": false // Disable help messages when command not found
}
```

### Add lazy-loaded CLI commands with `options.dir` and `app.commands`:

  Flatiron CLI will automatically lazy-load modules defining commands in the directory specified by `options.dir`. For example:

```js
// example2.js
var path = require('path'),
    flatiron = require('./lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli, {
  dir: path.join(__dirname, 'cmds')
});

app.start();
```

```js
// cmd/highfive.js
var highfive = module.exports = function highfive (person, cb) {
  this.log.info('High five to ' + person + '!');
  cb(null);
};
```

In the command, you expose a function of arguments and a callback. `this` is set to `app`, and the routing is taken care of automatically.

Here it is in action:

```
% node example2.js highfive Flatiron 
info:   High five to Flatiron!
```

You can also define these commands by adding them directly to `app.commands` yourself:

```
// example2b.js
var flatiron = require('./lib/flatiron'),
    app = flatiron.app;

var path = require('path'),
    flatiron = require('./lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli);

app.commands.highfive = function (person, cb) {
  this.log.info('High five to ' + person + '!');
  cb(null);
};

app.start();
```

```
% node example2b.js highfive Flatiron 
info:   High five to Flatiron!
```

Callback will always be the last argument provided to a function assigned to command

```js
app.commands.highfive = function (person, cb) {
  this.log.info('High five to ' + person + '!');
  console.log(arguments);
}
```

```
% node example2b.js highfive Flatiron lol haha
info:    High five to Flatiron!
{
  '0': 'Flatiron',
  '1': 'lol',
  '2': 'haha',
  '3': [Function]
}
```

### Define Ad-Hoc Commands With `app.cmd(path, handler)`:

This adds the cli routing path `path` to the app's CLI router, using the [director](https://github.com/flatiron/director) route handler `handler`, aliasing `app.router.on`. `cmd` routes are defined the same way as http routes, except that it uses ` ` (a space) for a delimiter instead of `/`.

For example:

```js
// example.js
var flatiron = require('./lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli, {
  usage: [
    'usage: node test.js hello <person>',
    '',
    '  This will print "hello <person>"'
  ]
});

app.cmd('hello :person', function (person) {
  app.log.info('hello ' + person + '!');
});

app.start()
```

When you run this program correctly, it will say hello:

```
% node example.js hello person
info:   hello person!
```

If not, you get a friendly usage message:

```
% node test.js hello
help:   usage: node test.js hello <person>
help:
help:     This will print "hello <person>"
```

### Check CLI Arguments with `app.argv`:

Once your app is started, `app.argv` will contain the [optimist](http://github.com/substack/node-optimist)-parsed argv options hash, ready to go!

Here's an example:

```js
// example3.js
var flatiron = require('./lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli);

app.start();

app.log.info(JSON.stringify(app.argv));
```

This prints:

```
% node example3.js
info:    {"_":[], "$0": "node ./example3.js"}
```

Awesome!

### Add a Default Help Command with `options.usage`:

When attaching the CLI plugin, just specify options.usage to get a friendly default message for when there aren't any matching routes:

```js
// example4.js
var flatiron = require('./lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli, {
  usage: [
    'Welcome to my app!',
    'Your command didn\'t do anything.',
    'This is expected.'
  ]
});

app.start();
```

```
% node example4.js 
help:   Welcome to my app!
help:   Your command didn't do anything.
help:   This is expected.
```

### Start The Application with `app.start(callback)`:

As seen in these examples, starting your app is as easy as `app.start`! this method takes a callback, which is called when an `app.command` completes. Here's a complete example demonstrating this behavior and how it integrates with `options.usage`:

```js
// example5.js
var path = require('path'),
    flatiron = require('./lib/flatiron'),
    app = flatiron.app;

app.use(flatiron.plugins.cli, {
  usage: [
    '`node example5.js error`: Throws an error.',
    '`node example5.js friendly`: Does not throw an error.'
  ]
});

app.commands.error = function (cb) {
  cb(new Error('I\'m an error!'));
};

app.commands.friendly = function (cb) {
  cb(null);
}

app.start(function (err) {
  if (err) {
    app.log.error(err.message || 'You didn\'t call any commands!');
    app.log.warn('NOT OK.');
    return process.exit(1);
  }
  app.log.info('OK.');
});
```

Here's how our app behaves:

```
% node example5.js friendly
info:   OK.

% node example5.js error
error:  I'm an error!
warn:   NOT OK.

% node example5.js
help:   `node example2b.js error`: Throws an error.
help:   `node example2b.js friendly`: Does not throw an error.
error:  You didn't call any commands!
warn:   NOT OK.
```

# Read More About Flatiron!

## Articles

* [Scaling Isomorphic Javascript Code](http://blog.nodejitsu.com/scaling-isomorphic-javascript-code)
* [Introducing Flatiron](http://blog.nodejitsu.com/introducing-flatiron)
* [Writing CLI Apps with Flatiron](http://blog.jit.su/writing-cli-apps-with-flatiron)

## Sub-Projects

* [Broadway](https://github.com/flatiron/broadway)
* [Union](https://github.com/flatiron/union)
* [Director](https://github.com/flatiron/director)
* [Plates](https://github.com/flatiron/plates)
* [Resourceful](https://github.com/flatiron/resourceful)
* [And More](https://github.com/flatiron)!

# Tests

Tests are written in vows:

``` bash
  $ npm test
```

#### Author: [Nodejitsu Inc.](http://nodejitsu.com)
#### License: MIT
