var temp = require('../lib/temp'),
    fs   = require('fs'),
    util = require('util'),
    path = require('path'),
    exec = require('child_process').exec;

var myData = "\\starttext\nHello World\n\\stoptext";

temp.mkdir('pdfcreator', function(err, dirPath) {
  var inputPath = path.join(dirPath, 'input.tex')
  fs.writeFile(inputPath, myData, function(err) {
    if (err) throw err;
    process.chdir(dirPath);
    exec("texexec '" + inputPath + "'", function(err) {
      if (err) throw err;
      fs.readFile(path.join(dirPath, 'input.pdf'), function(err, data) {
        if (err) throw err;
        util.print(data);
      });
    });
  });
});
