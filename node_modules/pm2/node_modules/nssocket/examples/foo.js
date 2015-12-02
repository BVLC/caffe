var nssocket = require('../lib/nssocket');
var outbound = new nssocket.NsSocket();
 
outbound.data('Broadcasting', function (data) {
  console.log(data)
});

outbound.connect(4949);

outbound.send('Connecting', { "random": Math.random() });
