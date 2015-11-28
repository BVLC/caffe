'use strict';

var cleanBaseURL = require('clean-base-url');
var debug        = require('debug')('ember-cli:serve-files');

function ServeFilesAddon(project) {
  this.project = project;
  this.name = 'serve-files-middleware';
}

ServeFilesAddon.prototype.serverMiddleware = function(options) {
  var app = options.app;
  options = options.options;
  debug('serverMiddleware: %s', options.outputPath);

  var broccoliMiddleware = options.middleware || require('broccoli/lib/middleware');
  var middleware = broccoliMiddleware(options.watcher, {
    liveReloadPath: '/ember-cli-live-reload.js',
    autoIndex: false // disable directory listings
  });

  var baseURL = cleanBaseURL(options.baseURL);

  debug('serverMiddleware: output: %s baseURL: %s', options.outputPath, baseURL);

  app.use(function ServeFiles(req, res, next) {
    var oldURL = req.url;
    var url = req.serveUrl || req.url;
    debug('serving: %s', url);

    var actualPrefix   = req.url.slice(0, baseURL.length - 1); // Don't care
    var expectedPrefix = baseURL.slice(0, baseURL.length - 1); // about last slash

    if (actualPrefix === expectedPrefix) {
      req.url = url.slice(actualPrefix.length); // Remove baseURL prefix
      debug('serving: (prefix stripped) %s', req.url);

      // Serve file, if no file has been found, reset url for proxy stuff
      // that comes afterwards
      middleware(req, res, function(err) {
        req.url = oldURL;
        if (err) {
          debug('err', err);
        }
        next(err);
      });
    } else {
      debug('prefixes didn\'t match, passing control on: (actual:%s expected:%s)', actualPrefix, expectedPrefix);
      next();
    }
  });
};

module.exports = ServeFilesAddon;
