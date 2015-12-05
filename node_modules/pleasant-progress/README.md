pleasant-progress
===================

module for outputing a pleasant progress indicator

Example
=======

```js
var progress = new PleasantProgress();

progress.start('building');

// => building .
// => building ..
// => building ...
// => building .

progress.stop();
```
