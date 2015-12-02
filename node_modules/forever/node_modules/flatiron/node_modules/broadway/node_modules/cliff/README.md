# cliff

CLI output formatting tools: "Your CLI Formatting Friend".

## Installation

### Installing npm (node package manager)
```
  curl http://npmjs.org/install.sh | sh
```

### Installing cliff
```
  [sudo] npm install cliff
```

## Usage
There are a number of methods available in Cliff for common logging tasks in command-line tools. If you're looking for more usage, checkout the [examples in this repository][3]:

1. Logging rows of data
2. Inspecting Objects

### Logging rows of data

**cliff.stringifyRows(rows[, colors])**

Takes a set of Arrays and row headers and returns properly formatted and padded rows. Here's a sample:

``` js
  var cliff = require('../lib/cliff');

  var rows = [
    ['Name',  'Flavor',    'Dessert'],
    ['Alice', 'cherry',    'yogurt'],
    ['Bob',   'carmel',    'apples'],
    ['Joe',   'chocolate', 'cake'],
    ['Nick',  'vanilla',   'ice cream']
  ];

  console.log(cliff.stringifyRows(rows, ['red', 'blue', 'green']));
```

![output from string-rows.js][string-rows]

**cliff.putRows(level, rows[, colors])**

The `putRows` method is a simple helper that takes a set of Arrays and row headers and logs properly formatted and padded rows (logs `stringifyRows` to [winston][0]). Here's a quick sample:

``` js
  var cliff = require('../lib/cliff');

  var rows = [
    ['Name',  'Flavor',    'Dessert'],
    ['Alice', 'cherry',    'yogurt'],
    ['Bob',   'carmel',    'apples'],
    ['Joe',   'chocolate', 'cake'],
    ['Nick',  'vanilla',   'ice cream']
  ];

  cliff.putRows('data', rows, ['red', 'blue', 'green']);
```

The resulting output on the command-line would be:

![output from put-rows.js][put-rows]

**cliff.stringifyObjectRows(objs, properties[, colors])**
*used to be: cliff.rowifyObjects(objs, properties, colors)*

Takes a set of Objects and the properties to extract from them and returns properly formatted and padded rows. Here's a sample:

``` js
  var cliff = require('../lib/cliff');

  var objs = [], obj = {
    name: "bazz",
    address: "1234 Nowhere Dr.",
  };

  for (var i = 0; i < 10; i++) {
    objs.push({
      name: obj.name,
      address: obj.address,
      id: Math.random().toString()
    });
  }

  console.log(cliff.stringifyObjectRows(objs, ['id', 'name', 'address'], ['red', 'blue', 'green']));
```

![output from string-object-rows.js][string-object-rows]

**cliff.putObjectRows(level, objs, properties[, colors])**

Takes a set of Objects and the properties to extract from them and it will log to the console. (it prints `stringifyObjectRows` with [winston][0]). Here's a sample:

``` js
  var cliff = require('../lib/cliff');

  var objs = [], obj = {
    name: "bazz",
    address: "1234 Nowhere Dr.",
  };

  for (var i = 0; i < 10; i++) {
    objs.push({
      name: obj.name,
      address: obj.address,
      id: Math.random().toString()
    });
  }

  cliff.putObjectRows('data', objs, ['id', 'name', 'address']);
```

![output from string-object-rows.js][string-object-rows]

**Colors Parameter**

The `colors` parameter is an array that colors the first row. It uses the [colors.js][2]. You can use any of those.

``` js
  var cliff = require('../lib/cliff');

  var rows = [
    ['Name',        'Flavor',              'Dessert'],
    ['Alice'.grey,  'cherry'.cyan,         'yogurt'.yellow],
    ['Bob'.magenta, 'carmel'.rainbow,      'apples'.white],
    ['Joe'.italic,  'chocolate'.underline, 'cake'.inverse],
    ['Nick'.bold,   'vanilla',             'ice cream']
  ];

  cliff.putRows('data', rows, ['red', 'blue', 'green']);
```

The resulting output on the command-line would be:

![output from puts-rows-colors.js][put-rows-colors]

### Inspecting Objects

**cliff.inspect(obj)**

The `inspect` method is a lightweight wrapper to a pre-configured [eyes][1] inspector. If you wish to change the coloring of objects that are logged using `cliff` you only need to override `cliff.inspect` with a new [eyes][1] inspector. Here is how to use it:

``` js
  var cliff = require('../lib/cliff');

  console.log(cliff.inspect({
    literal: "bazz",
    arr: [
      "one",
      2,
    ],
    obj: {
      host: "localhost",
      port: 5984,
      auth: {
        username: "admin",
        password: "password"
      }
    }
  }));
```

![output from inspect.js][inspect]

**cliff.putObject(obj, [rewriters, padding])**

The `putObject` method is a simple helper function for prefixing and styling inspected object output from [eyes][1]. Here's a quick sample:

``` js
var cliff = require('cliff');

cliff.putObject({
  literal: "bazz",
  arr: [
    "one",
    2,
  ],
  obj: {
    host: "localhost",
    port: 5984,
    auth: {
      username: "admin",
      password: "password"
    }
  }
});
```

The resulting output on the command-line would be: 

![output from put-object.js][put-object]

## Run Tests

All of the cliff tests are written in [vows][4], and cover all of the use cases described above.

```
  npm test
```

## Motivation

Cliff is the swiss army knife of CLI formatting tools. It is based on highly flexible and powerful libraries: 

* [winston][0]: A multi-transport async logging library for node.js
* [eyes][1]: A customizable value inspector for node.js
* [colors][2]: Get colors in your node.js console like what 


#### Author: [Charlie Robbins](http://twitter.com/indexzero)

[0]: http://github.com/indexzero/winston
[1]: http://github.com/cloudhead/eyes.js
[2]: http://github.com/marak/colors.js
[3]: http://github.com/flatiron/cliff/tree/master/examples
[4]: http://vowsjs.org

[inspect]: https://github.com/flatiron/cliff/raw/master/assets/inspect.png
[put-object-rows]: https://github.com/flatiron/cliff/raw/master/assets/put-object-rows.png
[put-object]: https://github.com/flatiron/cliff/raw/master/assets/put-object.png
[put-rows-colors]: https://github.com/flatiron/cliff/raw/master/assets/put-rows-colors.png
[put-rows]: https://github.com/flatiron/cliff/raw/master/assets/put-rows.png
[string-object-rows]: https://github.com/flatiron/cliff/raw/master/assets/string-object-rows.png
[string-rows]: https://github.com/flatiron/cliff/raw/master/assets/string-rows.png