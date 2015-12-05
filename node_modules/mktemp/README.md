# mktemp

[![Build Status](https://travis-ci.org/sasaplus1/mktemp.svg)](https://travis-ci.org/sasaplus1/mktemp)
[![Dependency Status](https://gemnasium.com/sasaplus1/mktemp.svg)](https://gemnasium.com/sasaplus1/mktemp)
[![NPM version](https://badge.fury.io/js/mktemp.svg)](http://badge.fury.io/js/mktemp)

mktemp command for node.js

## Installation

```sh
$ npm install mktemp
```

## Usage

```js
var mktemp = require('mktemp');

mktemp.createFile('XXXXX.txt', function(err, path) {
  if (err) throw err;

  // path match a /^[\da-zA-Z]{5}\.txt$/
  console.log(path);
});

// return value match a /^[\da-zA-Z]{5}\.tmp$/
mktemp.createFileSync('XXXXX.tmp');

mktemp.createDir('XXXXXXX', function(err, path) {
  if (err) throw err;

  // path match a /^[\da-zA-Z]{7}$/
  console.log(path);
});

// return value match a /^XXX-[\da-zA-Z]{3}$/
mktemp.createDirSync('XXX-XXX');
```

mktemp functions are replace to random string from placeholder "X" in template. see example:

```js
mktemp.createFileSync('XXXXXXX');  // match a /^[\da-zA-Z]{7}$/
mktemp.createFileSync('XXX.tmp');  // match a /^[\da-zA-Z]{3}\.tmp$/
mktemp.createFileSync('XXX-XXX');  // match a /^XXX-[\da-zA-Z]{3}$/
```

## Functions

### createFile(template, callback)

* `template`
  * `String` - filename template
* `callback`
  * `function(err, path)` - callback function
    * `err` : `Error|Null` - error object
    * `path` :  `String` -  path

create blank file of unique filename.
permission is `0600`.

### createFileSync(template)

* `template`
  * `String` - filename template
* `return`
  * `String` - path

sync version createFile.

### createDir(template, callback)

* `template`
  * `String` - dirname template
* `callback`
  * `function(err, path)` - callback function
    * `err` : `Error|Null` - error object
    * `path` : `String` - path

create directory of unique dirname.
permission is `0700`.

### createDirSync(template)

* `template`
  * `String` - dirname template
* `return`
  * `String` - path

sync version createDir.

## Test

```sh
$ npm install
$ npm test
```

## Contributors

* [Michael Ficarra](https://github.com/michaelficarra)

## License

The MIT license. Please see LICENSE file.
