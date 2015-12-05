var charm = require('../')();
charm.pipe(process.stdout);

charm.write('Progress: 0 %');
var i = 0;

var iv = setInterval(function () {
    charm.left(i.toString().length + 2);
    i ++;
    charm.write(i + ' %');
    if (i === 100) {
        charm.end('\nDone!\n');
        clearInterval(iv);
    }
}, 25);

charm.on('^C',process.exit);

