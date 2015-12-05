# lodash-node v3.10.1

The [compatibility & modern builds](https://github.com/lodash/lodash/wiki/Build-Differences) of [lodash](https://lodash.com/) exported as [Node.js](http://nodejs.org/)/[io.js](https://iojs.org/) modules.

Generated using [lodash-cli](https://www.npmjs.com/package/lodash-cli):
```bash
$ lodash modularize compat exports=node -o ./compat && lodash compat exports=node -d -o ./compat/index.js
$ lodash modularize modern exports=node -o ./modern && lodash modern exports=node -d -o ./modern/index.js
```

## Deprecated

The `lodash-node` package is deprecated in favor of [lodash](https://www.npmjs.com/package/lodash) & [lodash-compat](https://www.npmjs.com/package/lodash-compat) ≥ v3.0.0.

## Installation

Using npm:

```bash
$ {sudo -H} npm i -g npm
$ npm i --save lodash-node
```

In Node.js/io.js:

```js
// load the modern build
var _ = require('lodash-node');
// or the compatibility build
var _ = require('lodash-node/compat');
// or a method category
var array = require('lodash-node/modern/array');
// or a method
var chunk = require('lodash-node/compat/array/chunk');
```

See the [package source](https://github.com/lodash/lodash-node/tree/3.10.1) for more details.

**Note:**<br>
Don’t assign values to the [special variable](http://nodejs.org/api/repl.html#repl_repl_features) `_` when in the REPL.<br>
Install [n_](https://www.npmjs.com/package/n_) for a REPL that includes lodash by default.
