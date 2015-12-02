
var axm = require('../..');

setInterval(function() {
  axm.emit('is object', {
    user : 'toto',
    subobj : {
      subobj : {
        a : 'b'
      }
    }
  });

  axm.emit('is string', 'HEY!');

  axm.emit('is bool', true);

}, 100);
