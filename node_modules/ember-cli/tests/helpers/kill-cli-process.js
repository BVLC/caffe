'use strict';

module.exports = function(childProcess) {
  if (process.platform === 'win32') {
    childProcess.send({kill: true});
  } else {
    childProcess.kill('SIGINT');
  }
};
