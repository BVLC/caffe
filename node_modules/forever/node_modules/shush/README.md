##### shush
Hush up those JSON comments.

`shush` is a simple node module that allows JSON files containing comments to be read into
a module using a `require`-like syntax.

```json
/* jsonWithComments.js */
{
    // a property
    "myProp": "isCool"
}
```
```javascript
// foo.js
var shush = require('shush'),
    config = shush('./jsonWithComments');

console.log(config); // {"myProp": "isCool"}
```

Forthcoming feature: streaming.