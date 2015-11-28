'use strict';

var chalk = require('chalk');

module.exports = function(message, test) {
  if(!test) { return; }

  console.log(chalk.yellow('DEPRECATION: ' + message));
};

module.exports.deprecateUI = function(ui){
  return function(message, test) {
    if(!test) { return; }

    ui.writeLine(chalk.yellow('DEPRECATION: ' + message));
  };
};
