var util = require('util'),
    path = require('path'),
    spawn = require('child_process').spawn;
    
var child = spawn('node', [path.join(__dirname, 'count-timer.js')], { cwd: __dirname });

child.stdout.on('data', function (data) {
  util.puts(data);
  //throw new Error('User generated fault.');
});

child.stderr.on('data', function (data) {
  util.puts(data);
});

child.on('exit', function (code) {
  util.puts('Child process exited with code: ' + code);
});
