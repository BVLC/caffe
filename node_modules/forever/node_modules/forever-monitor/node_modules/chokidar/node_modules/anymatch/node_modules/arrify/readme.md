# arrify [![Build Status](https://travis-ci.org/sindresorhus/arrify.svg?branch=master)](https://travis-ci.org/sindresorhus/arrify)

> Convert a value to an array


## Install

```
$ npm install --save arrify
```


## Usage

```js
var arrify = require('arrify');

arrify('unicorn');
//=> ['unicorn']

arrify(['unicorn']);
//=> ['unicorn']

arrify(null);
//=> []

arrift(undefined);
//=> []
```

*Supplying `null` or `undefined` results in an empty array.*


## License

MIT © [Sindre Sorhus](http://sindresorhus.com)
