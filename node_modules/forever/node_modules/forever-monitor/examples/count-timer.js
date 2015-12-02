var util = require('util');

var count = 0;

var id = setInterval(function () {
  util.puts('Count is ' + count + '. Incrementing now.');
  count++;
}, 1000);
