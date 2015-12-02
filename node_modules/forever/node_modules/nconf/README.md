# nconf [![Build Status](https://secure.travis-ci.org/flatiron/nconf.png)](http://travis-ci.org/flatiron/nconf)

Hierarchical node.js configuration with files, environment variables, command-line arguments, and atomic object merging.

## Example
Using nconf is easy; it is designed to be a simple key-value store with support for both local and remote storage. Keys are namespaced and delimited by `:`. Lets dive right into sample usage:

``` js
  var fs    = require('fs'),
      nconf = require('nconf');

  //
  // Setup nconf to use (in-order):
  //   1. Command-line arguments
  //   2. Environment variables
  //   3. A file located at 'path/to/config.json'
  //
  nconf.argv()
       .env()
       .file({ file: 'path/to/config.json' });

  //
  // Set a few variables on `nconf`.
  //
  nconf.set('database:host', '127.0.0.1');
  nconf.set('database:port', 5984);

  //
  // Get the entire database object from nconf. This will output
  // { host: '127.0.0.1', port: 5984 }
  //
  console.log('foo: ' + nconf.get('foo'));
  console.log('NODE_ENV: ' + nconf.get('NODE_ENV'));
  console.log('database: ' + nconf.get('database'));

  //
  // Save the configuration object to disk
  //
  nconf.save(function (err) {
    fs.readFile('path/to/your/config.json', function (err, data) {
      console.dir(JSON.parse(data.toString()))
    });
  });
```

If you run the above script:

``` bash
  $ NODE_ENV=production sample.js --foo bar
```

The output will be:

```
  foo: bar
  NODE_ENV: production
  database: { host: '127.0.0.1', port: 5984 }
```

## Hierarchical configuration

Configuration management can get complicated very quickly for even trivial applications running in production. `nconf` addresses this problem by enabling you to setup a hierarchy for different sources of configuration with no defaults. **The order in which you attach these configuration sources determines their priority in the hierarchy.** Lets take a look at the options available to you

  1. **nconf.argv(options)** Loads `process.argv` using optimist. If `options` is supplied it is passed along to optimist.
  2. **nconf.env(options)** Loads `process.env` into the hierarchy.
  3. **nconf.file(options)** Loads the configuration data at options.file into the hierarchy.
  4. **nconf.defaults(options)** Loads the data in options.store into the hierarchy.
  5. **nconf.overrides(options)** Loads the data in options.store into the hierarchy.

A sane default for this could be:

``` js
  var nconf = require('nconf');

  //
  // 1. any overrides
  //
  nconf.overrides({
    'always': 'be this value'
  });

  //
  // 2. `process.env`
  // 3. `process.argv`
  //
  nconf.env().argv();

  //
  // 4. Values in `config.json`
  //
  nconf.file('/path/to/config.json');

  //
  // Or with a custom name
  //
  nconf.file('custom', '/path/to/config.json');

  //
  // Or searching from a base directory.
  // Note: `name` is optional.
  //
  nconf.file(name, {
    file: 'config.json',
    dir: 'search/from/here',
    search: true
  });

  //
  // 5. Any default values
  //
  nconf.defaults({
    'if nothing else': 'use this value'
  });
```

## API Documentation

The top-level of `nconf` is an instance of the `nconf.Provider` abstracts this all for you into a simple API.

### nconf.add(name, options)
Adds a new store with the specified `name` and `options`. If `options.type` is not set, then `name` will be used instead:

``` js
  nconf.add('user', { type: 'file', file: '/path/to/userconf.json' });
  nconf.add('global', { type: 'file', file: '/path/to/globalconf.json' });
```

### nconf.use(name, options)
Similar to `nconf.add`, except that it can replace an existing store if new options are provided

``` js
  //
  // Load a file store onto nconf with the specified settings
  //
  nconf.use('file', { file: '/path/to/some/config-file.json' });

  //
  // Replace the file store with new settings
  //
  nconf.use('file', { file: 'path/to/a-new/config-file.json' });
```

### nconf.remove(name)
Removes the store with the specified `name.` The configuration stored at that level will no longer be used for lookup(s).

``` js
  nconf.remove('file');
```

## Storage Engines

### Memory
A simple in-memory storage engine that stores a nested JSON representation of the configuration. To use this engine, just call `.use()` with the appropriate arguments. All calls to `.get()`, `.set()`, `.clear()`, `.reset()` methods are synchronous since we are only dealing with an in-memory object.

``` js
  nconf.use('memory');
```

### Argv
Responsible for loading the values parsed from `process.argv` by `optimist` into the configuration hierarchy. See the [optimist option docs](https://github.com/substack/node-optimist/#optionskey-opt) for more on the option format.

``` js
  //
  // Can optionally also be an object literal to pass to `optimist`.
  //
  nconf.argv({
    "x": {
      alias: 'example',
      describe: 'Example description for usage generation',
      demand: true,
      default: 'some-value'
    }
  });
```

### Env
Responsible for loading the values parsed from `process.env` into the configuration hierarchy.

``` js
  //
  // Can optionally also be an Array of values to limit process.env to.
  //
  nconf.env(['only', 'load', 'these', 'values', 'from', 'process.env']);

  //
  // Can also specify a separator for nested keys (instead of the default ':')
  //
  nconf.env('__');
  // Get the value of the env variable 'database__host'
  var dbHost = nconf.get('database:host');

  //
  // Or use both options
  //
  nconf.env({
    separator: '__',
    whitelist: ['database__host', 'only', 'load', 'these', 'values']
  });
  var dbHost = nconf.get('database:host');
```

### Literal
Loads a given object literal into the configuration hierarchy. Both `nconf.defaults()` and `nconf.overrides()` use the Literal store.

``` js
  nconf.defaults({
    'some': 'default value'
  });
```

### File
Based on the Memory store, but provides additional methods `.save()` and `.load()` which allow you to read your configuration to and from file. As with the Memory store, all method calls are synchronous with the exception of `.save()` and `.load()` which take callback functions. It is important to note that setting keys in the File engine will not be persisted to disk until a call to `.save()` is made.

``` js
  nconf.file('path/to/your/config.json');
  // add multiple files, hierarchically. notice the unique key for each file
  nconf.file('user', 'path/to/your/user.json');
  nconf.file('global', 'path/to/your/global.json');
```

The file store is also extensible for multiple file formats, defaulting to `JSON`. To use a custom format, simply pass a format object to the `.use()` method. This object must have `.parse()` and `.stringify()` methods just like the native `JSON` object.

### Redis
There is a separate Redis-based store available through [nconf-redis][0]. To install and use this store simply:

``` bash
  $ npm install nconf
  $ npm install nconf-redis
```

Once installing both `nconf` and `nconf-redis`, you must require both modules to use the Redis store:

``` js
  var nconf = require('nconf');

  //
  // Requiring `nconf-redis` will extend the `nconf`
  // module.
  //
  require('nconf-redis');

  nconf.use('redis', { host: 'localhost', port: 6379, ttl: 60 * 60 * 1000 });
```

## Installation

### Installing npm (node package manager)
```
  curl http://npmjs.org/install.sh | sh
```

### Installing nconf
```
  [sudo] npm install nconf
```

## More Documentation
There is more documentation available through docco. I haven't gotten around to making a gh-pages branch so in the meantime if you clone the repository you can view the docs:

```
  open docs/nconf.html
```

## Run Tests
Tests are written in vows and give complete coverage of all APIs and storage engines.

``` bash
  $ npm test
```

#### Author: [Charlie Robbins](http://nodejitsu.com)
#### License: MIT

[0]: http://github.com/indexzero/nconf-redis
