#!/usr/bin/env node
var python = require('../lib/python').shell;
var mycallback = function(err, data) {
   if (err) {
     console.error(err);
   } else {
     process.stdout.write(data + '\n>>> ');
   }
};
process.stdout.write('Using Python from NodeJS\n>>> ');
process.stdin.resume();
process.stdin.setEncoding('utf8');
process.stdin.on('data', function (chunk) {
   python(chunk, mycallback);
});

process.stdin.on('end', function() {
   python('quit()');
});
