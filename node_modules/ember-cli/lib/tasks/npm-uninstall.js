'use strict';

// Runs `npm uninstall` in cwd

var NpmTask = require('./npm-task');

module.exports = NpmTask.extend({
  command: 'uninstall',
  startProgressMessage: 'Uninstalling packages for tooling via npm',
  completionMessage: 'Uninstalled packages for tooling via npm.'
});
