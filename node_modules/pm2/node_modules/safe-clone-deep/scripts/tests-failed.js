var path = require('path');
var open = require('open');
var cursor = require('ansi')(process.stdout);

cursor.write('\n\n\n')
  .brightRed().write('!!!!!')
  .brightYellow().write(' NOTICE: Unit tests or test coverage failed ')
  .brightRed().write('!!!!!\n\n')
  .fg.reset();

open(path.resolve(__dirname,'../coverage/lcov-report/index.html'));