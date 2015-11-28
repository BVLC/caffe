/*jshint node:true*/

var Blueprint = require('../../lib/models/blueprint');

module.exports = {
  description: 'Generates a relative proxy to another server.',

  anonymousOptions: [
    'local-path',
    'remote-url'
  ],

  locals: function(options) {
    var proxyUrl = options.args[2];
    return {
      path: '/' + options.entity.name.replace(/^\//, ''),
      proxyUrl: proxyUrl
    };
  },

  beforeInstall: function(options) {
    var serverBlueprint = Blueprint.lookup('server', {
      ui: this.ui,
      analytics: this.analytics,
      project: this.project
    });

    return serverBlueprint.install(options);
  },

  afterInstall: function() {
    return this.addPackagesToProject([
      { name: 'http-proxy', target: '^1.1.6' }
    ]);
  }
};
