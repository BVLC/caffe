node-python
===========

A super-simple wrapper for NodeJS to interact programatically with the Python shell. Enables the use of Python-based tools from Node.

[![NPM Stats](https://nodei.co/npm/python.png?downloads=true&stars=true)](https://npmjs.org/package/python)

![NPM Downloads](https://nodei.co/npm-dl/python.png?months=9)

Example
-------
This example starts a python child process, reads stdin for python commands, pipes them through to the python shell and runs the callback method with the resulting output. State is preserved in the shell between calls.

```javascript
// ------
// app.js
// ------
var python=require('python').shell;

// a callback to handle the response
var mycallback = function(err, data) {
   if (err) {
     console.error(err);
   } else {
     console.log("Callback function got : " + data);
   }
};

// to test, read and execute commands from stdin
process.stdin.resume();
process.stdin.setEncoding('utf8');
process.stdin.on('data', function(chunk) {
   python(chunk, mycallback);
});
```

License
-------
MIT
