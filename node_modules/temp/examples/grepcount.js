var temp = require('../lib/temp'),
    fs   = require('fs'),
    util = require('util'),
    exec = require('child_process').exec;

var myData = "foo\nbar\nfoo\nbaz";

temp.open('myprefix', function(err, info) {
  if (err) throw err;
  fs.write(info.fd, myData);
  fs.close(info.fd, function(err) {
    if (err) throw err;
    exec("grep foo '" + info.path + "' | wc -l", function(err, stdout) {
      if (err) throw err;
      util.puts(stdout.trim());
    });
  });
});
