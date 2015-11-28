'use strict';

function ProxyServerAddon(project) {
  this.project = project;
  this.name = 'proxy-server-middleware';
}

ProxyServerAddon.prototype.serverMiddleware = function(options) {
  var app = options.app, server = options.options.httpServer;
  options = options.options;

  if (options.proxy) {
    var proxy = require('http-proxy').createProxyServer({
      target: options.proxy,
      ws: true,
      secure: !options.insecureProxy,
      changeOrigin: true
    });

    proxy.on('error', function (e) {
      options.ui.writeLine('Error proxying to ' + options.proxy);
      options.ui.writeError(e);
    });

    var morgan  = require('morgan');

    options.ui.writeLine('Proxying to ' + options.proxy);

    server.on('upgrade', function (req, socket, head) {
      options.ui.writeLine('Proxying websocket to ' + req.url);
      proxy.ws(req, socket, head);
    });

    app.use(morgan('dev'));
    app.use(function(req, res) {
      return proxy.web(req, res);
    });
  }
};

module.exports = ProxyServerAddon;
