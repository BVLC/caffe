'use strict';

var Promise     = require('../../ext/promise');
var path        = require('path');
var fs          = require('fs');
var Task        = require('../../models/task');
var SilentError = require('silent-error');

function createServer(options) {
  var instance;

  var Server = (require('tiny-lr')).Server;
  Server.prototype.error = function() {
    instance.error.apply(instance, arguments);
  };
  instance = new Server(options);
  return instance;
}

module.exports = Task.extend({
  liveReloadServer: function(options) {
    if (this._liveReloadServer) {
      return this._liveReloadServer;
    }

    this._liveReloadServer = createServer(options);
    return this._liveReloadServer;
  },


  listen: function(options) {
    var server = this.liveReloadServer(options);

    return new Promise(function(resolve, reject) {
      server.error = reject;
      server.listen(options.port, options.host, resolve);
    });
  },

  start: function(options) {
    var tlroptions = {};

    tlroptions.ssl = options.ssl;
    tlroptions.host = options.liveReloadHost || options.host;
    tlroptions.port = options.liveReloadPort;

    if (options.liveReload !== true) {
      return Promise.resolve('Livereload server manually disabled.');
    }

    if (options.ssl) {
      tlroptions.key = fs.readFileSync(options.sslKey);
      tlroptions.cert = fs.readFileSync(options.sslCert);
    }

    // Reload on file changes
    this.watcher.on('change', this.didChange.bind(this));
    this.watcher.on('error', this.didChange.bind(this));

    // Reload on express server restarts
    this.expressServer.on('restart', this.didRestart.bind(this));

    var url = 'http' + (options.ssl ? 's' : '') + '://' + this.displayHost(tlroptions.host) + ':' + tlroptions.port;
    // Start LiveReload server
    return this.listen(tlroptions)
      .then(this.writeBanner.bind(this, url))
      .catch(this.writeErrorBanner.bind(this, url));
  },

  displayHost: function(specifiedHost) {
    return specifiedHost || 'localhost';
  },

  writeBanner: function(url) {
    this.ui.writeLine('Livereload server on ' + url);
  },

  writeErrorBanner: function(url) {
    throw new SilentError('Livereload failed on ' + url + '.  It is either in use or you do not have permission.');
  },

  didChange: function(results) {
    var filePath = path.relative(this.project.root, results.filePath || '');

    var canTrigger = this.project.liveReloadFilterPatterns.reduce(function(bool, pattern) {
      bool = bool && !filePath.match(pattern);
      return bool;
    }, true);

    if (canTrigger) {
      this.liveReloadServer().changed({
        body: {
          files: ['LiveReload files']
        }
      });

      this.analytics.track({
        name:    'broccoli watcher',
        message: 'live-reload'
      });
    }
  },

  didRestart: function() {
    this.liveReloadServer().changed({
      body: {
        files: ['LiveReload files']
      }
    });

    this.analytics.track({
      name:    'express server',
      message: 'live-reload'
    });
  }
});
