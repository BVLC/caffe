# is-object

[![build status][1]][2] [![dependency status][3]][4]

[![browser support][5]][6]

Checks whether a value is an object

Because `typeof null` is a troll.

## Example

```js
var isObject = require("is-object")

console.log(isObject(null)) // false
console.log(isObject(require("util"))) // true
```

## Installation

`npm install is-object`

## Contributors

 - Raynos

## MIT Licenced

  [1]: https://secure.travis-ci.org/Colingo/is-object.png
  [2]: http://travis-ci.org/Colingo/is-object
  [3]: http://david-dm.org/Colingo/is-object/status.png
  [4]: http://david-dm.org/Colingo/is-object
  [5]: http://ci.testling.com/Colingo/is-object.png
  [6]: http://ci.testling.com/Colingo/is-object
