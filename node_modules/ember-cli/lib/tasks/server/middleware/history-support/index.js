'use strict';

var path = require('path');
var fs   = require('fs');

var cleanBaseURL = require('clean-base-url');

function HistorySupportAddon(project) {
  this.project = project;
  this.name = 'history-support-middleware';
}

HistorySupportAddon.prototype.shouldAddMiddleware = function(environment) {
  var config = this.project.config(environment);
  var locationType = config.locationType;
  var historySupportMiddlewareEnabled = config.historySupportMiddleware;

  return ['auto', 'history'].indexOf(locationType) !== -1 || historySupportMiddlewareEnabled;
};

HistorySupportAddon.prototype.serverMiddleware = function(config) {
  if (this.shouldAddMiddleware(config.options.environment)) {
    this.addMiddleware(config);
  }
};

HistorySupportAddon.prototype.addMiddleware = function(config) {
  var app = config.app;
  var options = config.options;
  var watcher = options.watcher;

  var baseURL = cleanBaseURL(options.baseURL);
  var baseURLRegexp = new RegExp('^' + baseURL);

  app.use(function(req, res, next) {
    watcher.then(function(results) {

      var acceptHeaders = req.headers.accept || [];
      var hasHTMLHeader = acceptHeaders.indexOf('text/html') !== -1;
      var isForBaseURL = baseURLRegexp.test(req.path);

      if (hasHTMLHeader && isForBaseURL && req.method === 'GET') {
        var assetPath = req.path.slice(baseURL.length);
        var isFile = false;
        try { isFile = fs.statSync(path.join(results.directory, assetPath)).isFile(); } catch (err) { }
        if (!isFile) {
          req.serveUrl = baseURL + 'index.html';
        }
      }
    }).finally(next);
  });
};

module.exports = HistorySupportAddon;
