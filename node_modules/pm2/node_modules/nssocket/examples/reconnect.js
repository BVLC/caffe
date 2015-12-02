var net = require('net'),
    nssocket = require('../lib/nssocket');

net.createServer(function (socket) {
  //
  // Close the underlying socket after `1000ms`
  //
  setTimeout(function () {
    socket.destroy();
  }, 1000);
}).listen(8345);

//
// Create an NsSocket instance with reconnect enabled
//
var socket = new nssocket.NsSocket({
  reconnect: true,
  type: 'tcp4',
});

socket.on('start', function () {
  //
  // The socket will emit this event periodically
  // as it attempts to reconnect
  //
  console.dir('start');
});

socket.connect(8345);