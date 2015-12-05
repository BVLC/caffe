var charm = require('../')();
charm.pipe(process.stdout);

charm
    .column(16)
    .write('beep')
    .down()
    .column(32)
    .write('boop\n')
    .end()
;
