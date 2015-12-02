/*
 * static.js: Top-level plugin exposing st's static server to flatiron app
 *
 * (C) 2012, Nodejitsu, Inc.
 * MIT LICENSE
 *
 */

var path = require('path'),
    flatiron = require('../../flatiron'),
    common = flatiron.common, st;

try {
  //
  // Attempt to require st.
  //
  st = require('st');
}
catch (ex) {
  //
  // Do nothing since this is a progressive enhancement
  //
  console.warn('flatiron.plugins.static requires the `st` module from npm');
  console.warn('install using `npm install st`.');
  console.trace();
  process.exit(1);
}

exports.name = 'static';

exports.attach = function (options) {
  var app = this;

  options = options || {};

  //
  // Accept string `options`
  //
  if (typeof options === 'string') {
    options = { root: options };
  }

  //
  // Default overrides
  //
  options.passthrough = true;

  //
  // Url for static server
  //
  options.index = options.index || false;
  options.dot = options.dot || false;
  options.url = options.url || '/';

  //
  // Attempt to merge defaults passed to `app.use(flatiron.plugins.static)`
  // with any additional configuration that may have been loaded
  options = common.mixin(
    {},
    options,
    app.config.get('static') || {}
  );

  app.config.set('static', options);

  //
  // `app.static` api to be used by other plugins
  // to server static files
  //
  app.static = function (dir) {
    options.path = dir;
    app.http.before = app.http.before.concat(st(options));
  }

  // * `options.dir`: Explicit path to assets directory
  // * `options.root`: Relative root to the assets directory ('/app/assets')
  // * `app.root`: Relative root to the assets directory ('/app/assets')
  if (options.dir || options.root || app.root) {
    app._staticDir = options.dir
      || path.join(options.root || app.root, 'app', 'assets');

    //
    // Serve staticDir using middleware in union
    //
    app.static(app._staticDir);
  }
}
