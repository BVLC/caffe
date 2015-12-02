#### caller

Figure out your caller (thanks to @substack).

##### Initialization Time Caller
```javascript
// foo.js

var bar = require('bar');
```

```javascript
// bar.js

var caller = require('caller');
console.log(caller()); // `/path/to/foo.js`
```

##### Runtime Caller
```javascript
// foo.js

var bar = require('bar');
bar.doWork();
```

```javascript
// bar.js

var caller = require('caller');

exports.doWork = function () {
    console.log(caller());  // `/path/to/foo.js`
};
```