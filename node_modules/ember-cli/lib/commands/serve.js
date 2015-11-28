'use strict';

var assign      = require('lodash/object/assign');
var path        = require('path');
var Command     = require('../models/command');
var Promise     = require('../ext/promise');
var SilentError = require('silent-error');
var PortFinder  = require('portfinder');
var win         = require('../utilities/windows-admin');
var EOL         = require('os').EOL;

PortFinder.basePort = 49152;

var getPort = Promise.denodeify(PortFinder.getPort);
var defaultPort = process.env.PORT || 4200;

module.exports = Command.extend({
  name: 'serve',
  description: 'Builds and serves your app, rebuilding on file changes.',
  aliases: ['server', 's'],

  availableOptions: [
    { name: 'port',             type: Number,  default: defaultPort,   aliases: ['p'] },
    { name: 'host',             type: String,                          aliases: ['H'],     description: 'Listens on all interfaces by default' },
    { name: 'proxy',            type: String,                          aliases: ['pr', 'pxy'] },
    { name: 'insecure-proxy',   type: Boolean, default: false,         aliases: ['inspr'], description: 'Set false to proxy self-signed SSL certificates' },
    { name: 'watcher',          type: String,  default: 'events',      aliases: ['w'] },
    { name: 'live-reload',      type: Boolean, default: true,          aliases: ['lr'] },
    { name: 'live-reload-host', type: String,                          aliases: ['lrh'],   description: 'Defaults to host' },
    { name: 'live-reload-port', type: Number,                          aliases: ['lrp'],   description: '(Defaults to port number within [49152...65535])' },
    { name: 'environment',      type: String,  default: 'development', aliases: ['e', { 'dev': 'development' }, { 'prod': 'production' }] },
    { name: 'output-path',      type: path,    default: 'dist/',       aliases: ['op', 'out'] },
    { name: 'ssl',              type: Boolean, default: false },
    { name: 'ssl-key',          type: String,  default: 'ssl/server.key' },
    { name: 'ssl-cert',         type: String,  default: 'ssl/server.crt' }
  ],

  run: function(commandOptions) {
    var port = commandOptions.port ? Promise.resolve(commandOptions.port) : getPort({ host: commandOptions.host });
    var liveReloadHost = commandOptions.liveReloadHost || commandOptions.host;
    var liveReloadPort = commandOptions.liveReloadPort ? Promise.resolve(commandOptions.liveReloadPort) : getPort({ host: liveReloadHost });

    return Promise.all([liveReloadPort, port]).then(function(values) {
      var liveReloadPort = values[0];
      var port = values[1];
      commandOptions = assign({}, commandOptions, {
        port: port,
        liveReloadPort: liveReloadPort,
        liveReloadHost: liveReloadHost,
        baseURL: this.project.config(commandOptions.environment).baseURL || '/'
      });

      if (commandOptions.proxy) {
        if (!commandOptions.proxy.match(/^(http:|https:)/)) {
          var message = 'You need to include a protocol with the proxy URL.' + EOL + 'Try --proxy http://' + commandOptions.proxy;

          return Promise.reject(new SilentError(message));
        }
      }

      var ServeTask = this.tasks.Serve;
      var serve = new ServeTask({
        ui: this.ui,
        analytics: this.analytics,
        project: this.project
      });

      return win.checkWindowsElevation(this.ui).then(function() {
        return serve.run(commandOptions);
      });
    }.bind(this));
  }
});
