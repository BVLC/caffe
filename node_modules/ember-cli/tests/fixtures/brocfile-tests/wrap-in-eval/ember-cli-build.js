/* global require, module */

var EmberApp = require('ember-cli/lib/broccoli/ember-app');

module.exports = function(defaults) {
  var app = new EmberApp(defaults, {
    name: require('./package.json').name,
    wrapInEval: true,
    getEnvJSON: require('./config/environment')
  });

  return app.toTree();
};
