# broadway [![Build Status](https://secure.travis-ci.org/flatiron/broadway.png)](http://travis-ci.org/flatiron/broadway)

*Lightweight application extensibility and composition with a twist of feature
reflection.*

## Example

### app.js
```js
var broadway = require("broadway");

var app = new broadway.App();

// Passes the second argument to `helloworld.attach`.
app.use(require("./plugins/helloworld"), { "delimiter": "!" } );

app.init(function (err) {
  if (err) {
    console.log(err);
  }
});

app.hello("world");
```

### plugins/helloworld.js

```js
// `exports.attach` gets called by broadway on `app.use`
exports.attach = function (options) {

  this.hello = function (world) {
    console.log("Hello "+ world + options.delimiter || ".");
  };

};

// `exports.init` gets called by broadway on `app.init`.
exports.init = function (done) {

  // This plugin doesn't require any initialization step.
  return done();

};
```

### run it!

```bash
josh@onix:~/dev/broadway/examples$ node simple/app.js 
Hello world!
josh@onix:~/dev/broadway/examples$ 
```

## Installation

### Installing npm (node package manager)
``` bash
  $ curl http://npmjs.org/install.sh | sh
```

### Installing broadway
``` bash 
  $ [sudo] npm install broadway
```

## API

### App#init(callback)

Initialize application and it's plugins, `callback` will be called with null or
initialization error as first argument.

### App#use(plugin, options)

Attach plugin to application. `plugin` should conform to following interface:

```javascript
var plugin = {
  "name": "example-plugin", // Plugin's name

  "attach": function attach(options) {
    // Called with plugin options once plugin attached to application
    // `this` - is a reference to application
  },

  "detach": function detach() {
    // Called when plugin detached from application
    // (Only if plugin with same name was attached)
    // `this` - is a reference to application
  },

  "init": function init(callback) {
    // Called on application initialization
    // App#init(callback) will be called once every plugin will call `callback`
    // `this` - is a reference to application
  }
};
```

### App#on(event, callback) and App#emit(event, data)

App inherits from [EventEmitter2][2], and many plugins build on this
functionality.

#### Built-In Events:

* `error:init`: Broadway emits this event when it throws an error while attempting to initialize.

Read the [EventEmitter2][2] documentation for more information.

## Tests
All tests are written with [vows][0] and should be run with [npm][1]:

``` bash
  $ npm test
```

#### [Charlie Robbins](http://nodejitsu.com)
#### License: MIT

[0]: http://vowsjs.org
[1]: http://npmjs.org
[2]: https://github.com/hij1nx/EventEmitter2
