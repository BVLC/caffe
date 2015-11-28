var EmberApp = require('ember-cli/lib/broccoli/ember-app');
var app = new EmberApp();

app.import('vendor/brocfile-script.js');

module.exports = app.toTree();
