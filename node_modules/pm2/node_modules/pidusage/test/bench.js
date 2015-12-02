
var pidusage = require('.');

setInterval(function() {
  pidusage.stat(18932, function(err, res) {
    console.log(res);
  });
  console.log(pidusage._history);
}, 1000);
