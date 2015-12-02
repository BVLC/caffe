# prettyjson [![Build Status](https://secure.travis-ci.org/rafeca/prettyjson.png)](http://travis-ci.org/rafeca/prettyjson) [![NPM version](https://badge.fury.io/js/prettyjson.png)](http://badge.fury.io/js/prettyjson) [![Coverage Status](https://coveralls.io/repos/rafeca/prettyjson/badge.png?branch=master)](https://coveralls.io/r/rafeca/prettyjson?branch=master)

Package for formatting JSON data in a coloured YAML-style, perfect for CLI output.

## How to install

Just install it via NPM:

```bash
$ npm install -g prettyjson
```

This will install `prettyjson` globally, so it will be added automatically
to your `PATH`.

## Using it (from the CLI)

This package installs a command line interface to render JSON data in a more
convenient way. You can use the CLI in three different ways:

**Decode a JSON file:** If you want to see the contents of a JSON file, just pass
it as the first argument to the CLI:

```bash
$ prettyjson package.json
```

![Example 1](https://raw.github.com/rafeca/prettyjson/master/images/example3.png)

**Decode the stdin:** You can also pipe the result of a command (for example an
HTTP request) to the CLI to see the JSON result in a clearer way:

```bash
$ curl https://api.github.com/users/rafeca | prettyjson
```

![Example 2](https://raw.github.com/rafeca/prettyjson/master/images/example4.png)

**Decode random strings:** if you call the CLI with no arguments, you'll get a
prompt where you can past JSON strings and they'll be automatically displayed in a clearer way:

![Example 3](https://raw.github.com/rafeca/prettyjson/master/images/example5.png)

### Command line options

It's possible to customize the output through some command line options:

```bash
# Change colors
$ prettyjson --string=red --keys=blue --dash=yellow --number=green package.json

# Do not use colors
$ prettyjson --nocolor=1 package.json

# Change indentation
$ prettyjson --indent=4 package.json

# Render arrays elements in a single line
$ prettyjson --inline-arrays=1 package.json
```

**Deprecation Notice**: The old configuration through environment variables is
deprecated and it will be removed in the next major version (1.0.0).

## Using it (from Node.js)

It's pretty easy to use it. You just have to include it in your script and call
the `render()` method:

```javascript
var prettyjson = require('prettyjson');

var data = {
  username: 'rafeca',
  url: 'https://github.com/rafeca',
  twitter_account: 'https://twitter.com/rafeca',
  projects: ['prettyprint', 'connfu']
};

var options = {
  noColor: true
};

console.log(prettyjson.render(data, options));
```

And will output:

![Example 4](https://raw.github.com/rafeca/prettyjson/master/images/example1.png)

You can also configure the colors of the hash keys and array dashes
(using [colors.js](https://github.com/Marak/colors.js) colors syntax):

```javascript
var prettyjson = require('prettyjson');

var data = {
  username: 'rafeca',
  url: 'https://github.com/rafeca',
  twitter_account: 'https://twitter.com/rafeca',
  projects: ['prettyprint', 'connfu']
};

console.log(prettyjson.render(data, {
  keysColor: 'rainbow',
  dashColor: 'magenta',
  stringColor: 'white'
}));
```

Will output something like:

![Example 5](https://raw.github.com/rafeca/prettyjson/master/images/example2.png)

## Running Tests

To run the test suite first invoke the following command within the repo,
installing the development dependencies:

```bash
$ npm install
```

then run the tests:

```bash
$ npm test
```

On windows, you can run the tests with:

```cmd
C:\git\prettyjson> npm run-script testwin
```
