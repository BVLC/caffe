'use strict';

var http = require('http')        // (or https / spdy)
  , connect = require('connect')  // or express
  , nowww = require('./')
  , app = connect()
  , server
  ;

app
  .use(nowww())
  .use(require('serve-static')(__dirname + '/public/'))
  ;

server = http.createServer();
server.on('request', app);
server.listen(3000, function () {
  console.log('Listening on ' + server.address().port);
});
