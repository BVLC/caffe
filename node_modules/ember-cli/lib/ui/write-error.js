'use strict';
var chalk = require('chalk');

module.exports = function writeError(ui, error){
  if (!error) { return; }

  // Uglify errors have a filename instead
  var fileName = error.file || error.filename;
  if (fileName) {
    if (error.line) {
      fileName += error.col ? ' (' + error.line + ':' + error.col + ')' : ' (' + error.line + ')';
    }
    ui.writeLine(chalk.red('File: ' + fileName), 'ERROR');
  }

  if (error.message) {
    ui.writeLine(chalk.red(error.message), 'ERROR');
  } else {
    ui.writeLine(chalk.red(error), 'ERROR');
  }

  if (error.stack) {
    ui.writeLine(error.stack, 'ERROR');
  }
};
