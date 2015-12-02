var util = require('util'),
    path = require('path'),
    forever = require('./../lib/forever'),
    script = path.join(__dirname, 'server.js');
        
var child1 = new (forever.Monitor)(script, { 'options': [ "--port=8080"] });
child1.start();
util.puts('Forever process running server.js on 8080');

var child2 = new (forever.Monitor)(script, { 'options': [ "--port=8081"] });          
child2.start();
util.puts('Forever process running server.js on 8081');
