# blank-object

Object.create(null) turns out to be quite slow to alloc in v8, but instead if
we inherit from an ancestory with `proto = create(null)` we have nearly
the same functionallity but with dramatically faster alloc.

```js
var BlankObject = require('blank-object');

var bo = new BlankObject();
```

Every key is `undefined` but `"constructor" in blank` will return true.  This is designed for a presence check `map[key] !== undefined` since `in` is also slow like `hasOwnProperty`, `delete` and `Object.create`.

```js
function UNDEFINED() {}
export default class Map {
  constructor() {
    this.store = new BlankObject();
  }

  has(key) {
    return this.store[key] !== undefined;
  }

  get(key) {
    let val = this.store[key];
    return val === UNDEFINED ? undefined : val;
  }

  set(key, val) {
    this.store[key] = val === undefined ? UNDEFINED : val;
  }
}
```
