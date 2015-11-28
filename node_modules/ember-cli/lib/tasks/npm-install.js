'use strict';

// Runs `npm install` in cwd

var NpmTask = require('./npm-task');

module.exports = NpmTask.extend({
  command: 'install',
  startProgressMessage: 'Installing packages for tooling via npm',
  completionMessage: 'Installed packages for tooling via npm.'
});
