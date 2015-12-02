var util = require('util');

setTimeout(function () {
  util.puts('Throwing error now.');
  throw new Error('User generated fault.');
}, 200);
