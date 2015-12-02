var http = require('http');
var ecstatic = require('ecstatic')(__dirname);
var server = http.createServer(ecstatic);
server.listen(8000);
