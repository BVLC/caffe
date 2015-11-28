'use strict';

var path          = require('path');
var EventEmitter  = require('events').EventEmitter;
var chalk         = require('chalk');
var fs            = require('fs');
var existsSync    = require('exists-sync');
var debounce      = require('lodash/function/debounce');
var mapSeries     = require('promise-map-series');
var Promise       = require('../../ext/promise');
var Task          = require('../../models/task');
var SilentError   = require('silent-error');

var cleanBaseURL = require('clean-base-url');

module.exports = Task.extend({
  init: function() {
    this.emitter = new EventEmitter();
    this.express = this.express || require('express');
    this.http  = this.http  || require('http');
    this.https = this.https || require('https');

    var serverRestartDelayTime = this.serverRestartDelayTime || 100;
    this.scheduleServerRestart = debounce(function(){
      this.restartHttpServer();
    }, serverRestartDelayTime);
  },

  on: function() {
    this.emitter.on.apply(this.emitter, arguments);
  },

  off: function() {
    this.emitter.off.apply(this.emitter, arguments);
  },

  emit: function() {
    this.emitter.emit.apply(this.emitter, arguments);
  },

  displayHost: function(specifiedHost) {
    return specifiedHost || 'localhost';
  },

  setupHttpServer: function() {
    if (this.startOptions.ssl) {
      this.httpServer = this.createHttpsServer();
    } else {
      this.httpServer = this.http.createServer(this.app);
    }

    // We have to keep track of sockets so that we can close them
    // when we need to restart.
    this.sockets = {};
    this.nextSocketId = 0;
    this.httpServer.on('connection', function(socket) {
      var socketId = this.nextSocketId++;
      this.sockets[socketId] = socket;

      socket.on('close', function() {
        delete this.sockets[socketId];
      }.bind(this));
    }.bind(this));
  },

  createHttpsServer: function() {
    if(!existsSync(this.startOptions.sslKey)) {
      throw new TypeError('SSL key couldn\'t be found in "' + this.startOptions.sslKey + '", please provide a path to an existing ssl key file with --ssl-key');
    }
    if(!existsSync(this.startOptions.sslCert)) {
      throw new TypeError('SSL certificate couldn\'t be found in "' + this.startOptions.sslCert + '", please provide a path to an existing ssl certificate file with --ssl-cert');
    }
    var options = {
      key: fs.readFileSync(this.startOptions.sslKey),
      cert: fs.readFileSync(this.startOptions.sslCert)
    };
    return this.https.createServer(options, this.app);
  },

  listen: function(port, host) {
    var server = this.httpServer;
    return new Promise(function(resolve, reject) {
      server.listen(port, host);
      server.on('listening', function() {
        resolve();
        this.emit('listening');
      }.bind(this));
      server.on('error', reject);
    }.bind(this));
  },

  processAddonMiddlewares: function(options) {
    this.project.initializeAddons();
    return mapSeries(this.project.addons, function(addon) {
      if (addon.serverMiddleware) {
        return addon.serverMiddleware({
          app: this.app,
          options: options
        });
      }
    }, this);
  },

  processAppMiddlewares: function(options) {
    if (this.project.has(this.serverRoot)) {
      var server = this.project.require(this.serverRoot);
      if (typeof server !== 'function') {
        throw new TypeError('ember-cli expected ./server/index.js to be the entry for your mock or proxy server');
      }
      if (server.length === 3) {
        // express app is function of form req, res, next
        return this.app.use(server);
      }
      return server(this.app, options);
    }
  },

  start: function(options) {
    options.project       = this.project;
    options.watcher       = this.watcher;
    options.serverWatcher = this.serverWatcher;
    options.ui            = this.ui;

    this.startOptions = options;

    if (this.serverWatcher) {
      this.serverWatcher.on('change', this.serverWatcherDidChange.bind(this));
      this.serverWatcher.on('add', this.serverWatcherDidChange.bind(this));
      this.serverWatcher.on('delete', this.serverWatcherDidChange.bind(this));
    }

    return this.startHttpServer()
      .then(function () {
        var baseURL = cleanBaseURL(options.baseURL);

        options.ui.writeLine('Serving on http' + (options.ssl ? 's' : '') + '://' + this.displayHost(options.host) + ':' + options.port + baseURL);
      }.bind(this));
  },

  serverWatcherDidChange: function() {
    this.scheduleServerRestart();
  },

  restartHttpServer: function() {
    if (!this.serverRestartPromise) {
      this.serverRestartPromise =
        this.stopHttpServer()
          .then(function () {
            this.invalidateCache(this.serverRoot);
            return this.startHttpServer();
          }.bind(this))
          .then(function () {
            this.emit('restart');
            this.ui.writeLine('');
            this.ui.writeLine(chalk.green('Server restarted.'));
            this.ui.writeLine('');
          }.bind(this))
          .catch(function (err) {
            this.ui.writeError(err);
          }.bind(this))
          .finally(function () {
            this.serverRestartPromise = null;
          }.bind(this));
      return this.serverRestartPromise;
    } else {
      return this.serverRestartPromise.then(function () {
        return this.restartHttpServer();
      }.bind(this));
    }
  },

  stopHttpServer: function() {
    return new Promise(function (resolve, reject) {
      if (!this.httpServer) {
        return resolve();
      }
      this.httpServer.close(function (err) {
        if (err) {
          reject(err);
          return;
        }
        this.httpServer = null;
        resolve();
      }.bind(this));

      // We have to force close all sockets in order to get an fast restart
      var sockets = this.sockets;
      for (var socketId in sockets) {
        sockets[socketId].destroy();
      }
    }.bind(this));
  },

  startHttpServer: function() {
    this.app = this.express();
    this.app.use(require('compression')());

    this.setupHttpServer();

    var options = this.startOptions;
    options.httpServer = this.httpServer;

    return Promise.resolve()
      .then(function(){
        return this.processAppMiddlewares(options);
      }.bind(this))
      .then(function(){
        return this.processAddonMiddlewares(options);
      }.bind(this))
      .then(function(){
        return this.listen(options.port, options.host)
          .catch(function() {
            throw new SilentError('Could not serve on http://' + this.displayHost(options.host) + ':' + options.port + '. It is either in use or you do not have permission.');
          }.bind(this));
      }.bind(this));
  },

  invalidateCache: function (serverRoot) {
    var absoluteServerRoot = path.resolve(serverRoot);
    if (absoluteServerRoot[absoluteServerRoot.length - 1] !== path.sep) {
      absoluteServerRoot += path.sep;
    }

    var allKeys = Object.keys(require.cache);
    for (var i = 0; i < allKeys.length; i++) {
      if (allKeys[i].indexOf(absoluteServerRoot) === 0) {
        delete require.cache[allKeys[i]];
      }
    }
  }
});
