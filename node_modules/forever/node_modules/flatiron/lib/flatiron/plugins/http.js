/*
 * http.js: Top-level plugin exposing HTTP features in flatiron
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var director = require('director'),
    flatiron = require('../../flatiron'),
    union;

try {
  //
  // Attempt to require union.
  //
  union = require('union');
}
catch (ex) {
  //
  // Do nothing since this is a progressive enhancement
  //
  console.warn('flatiron.plugins.http requires the `union` module from npm');
  console.warn('install using `npm install union`.');
  console.trace();
  process.exit(1);
}

//
// Name this plugin.
//
exports.name = 'http';

exports.attach = function (options) {
  var app = this;

  //
  // Define the `http` namespace on the app for later use
  //
  app.http = app.http || {};

  app.http = flatiron.common.mixin({}, app.http, options || {});

  app.http.route = flatiron.common.mixin({ async: true }, app.http.route || {});

  app.http.buffer = (app.http.buffer !== false);
  app.http.before = app.http.before || [];
  app.http.after = app.http.after || [];
  app.http.headers = app.http.headers || {
    'x-powered-by': 'flatiron ' + flatiron.version
  };

  app.router = new director.http.Router().configure(app.http.route);

  app.start = function (port, host, callback) {
    if (!callback && typeof host === 'function') {
      callback = host;
      host = null;
    }

    app.createServer();

    app.init(function (err) {
      if (err) {
        if (callback) {
          return callback(err);
        }

        throw err;
      }

      app.listen(port, host, callback);
    });
  };

  app.createServer = function(){
    app.server = union.createServer({
      buffer: app.http.buffer,
      after: app.http.after,
      before: app.http.before.concat(function (req, res) {
        if (!app.router.dispatch(req, res, app.http.onError || union.errorHandler)) {
          if (!app.http.onError) res.emit('next');
        }
      }),
      headers: app.http.headers,
      limit: app.http.limit,
      https: app.http.https,
      spdy: app.http.spdy
    });
  };

  app.listen = function (port, host, callback) {
    if (!callback && typeof host === 'function') {
      callback = host;
      host = null;
    }

    if (!app.server) app.createServer();

    return host
      ? app.server.listen(port, host, callback)
      : app.server.listen(port, callback);
  };
};
