# get [![Build Status](https://secure.travis-ci.org/mapbox/node-get.png?branch=master)](http://travis-ci.org/mapbox/node-get)

`get` is a slightly higher-level HTTP client for nodejs.

## Installation

    npm install get

get has no dependencies.

For testing, you'll need make and [mocha](https://github.com/visionmedia/mocha).

For docs you'll need [docco](https://github.com/jashkenas/docco).

## Features

* Redirect following.
* Convenience functions for downloading and getting data as string.
* Binary-extension and basic binary detection.
* Configurable headers

## API

Downloads are objects in `get`.

```javascript
var dl = new get({ uri: 'http://google.com/' });
```

However, the function is [a self-calling constructor](http://ejohn.org/blog/simple-class-instantiation/), and thus the `new` keyword is not necessary:

```javascript
var dl = get({ uri: 'http://google.com/' });
```

The get constructor can also take a plain string if you don't want to give options.

```javascript
var dl = get('http://google.com/');
```

It can also take other options.

```javascript
var dl = get({
    uri: 'http://google.com/',
    max_redirs: 20,
});
```

Then it exposes three main methods

```javascript
dl.asString(function(err, str) {
    console.log(str);
});
```

and

```javascript
dl.toDisk('myfile.txt', function(err, filename) {
    console.log(err);
});
```

and finally

```javascript
dl.asBuffer(function(err, data) {
    console.log(data);
});
```


There's also a lower-level API.

```javascript
dl.perform(function(err, response) {
    // response is just a response object, just like
    // HTTP request, except handling redirects
});
```

If you give node-get an object of settings instead of a string,
it accepts

* `uri` - the address of the resource
* `headers` - to replace its default headers with custom ones
* `max_redirs` - the number of redirects to follow before returning an error
* `no_proxy` - don't use a HTTP proxy, even if one is in `ENV`
* `encoding` - When calling `.guessEncoding()`, `get` will use this instead of the default value

## Example

```
var get = require('get');

get('http://google.com/').asString(function(err, data) {
    if (err) throw err;
    console.log(data);
});
```

## TODO:

* Guessing encoding wth headers
* User-customizable encodings

## Authors

* Tom MacWright (tmcw)
* Konstantin Kaefer (kkaefer)
