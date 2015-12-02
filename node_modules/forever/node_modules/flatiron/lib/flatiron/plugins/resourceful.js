/*
 * resourceful.js: Top-level plugin exposing resourceful to flatiron app
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
var path = require('path'),
    fs = require('fs'),
    flatiron = require('../../flatiron'),
    common = flatiron.common,
    resourceful,
    existsSync = fs.existsSync || path.existsSync;

try {
  //
  // Attempt to require resourceful.
  //
  resourceful = require('resourceful');
}
catch (ex) {
  //
  // Do nothing since this is a progressive enhancement
  //
  console.warn('flatiron.plugins.resourceful requires the `resourceful` module from npm');
  console.warn('install using `npm install resourceful`.');
  console.trace();
  process.exit(1);
}

exports.name = 'resourceful';

exports.attach = function (options) {
  var app = this;
  options = options || {};

  //
  // Accept string `options`.
  //
  if (typeof options === 'string') {
    options = { root: options };
  }

  //
  // Create `app.resources` if it does not exist already.
  //
  app.resources = app.resources || {};

  //
  // Expose a couple of resourceful helpers
  //
  app.define = resourceful.define;

  //
  // Lazy-load the resources directory based on a few intelligent defaults:
  //
  // * `options.dir`: Explicit path to resources directory
  // * `options.root`: Relative root to the resources directory ('/app/resources')
  // * `app.root`: Relative root to the resources directory ('/app/resources')
  //
  if (options.dir || options.root || app.root) {
    app._resourceDir = options.dir
      || path.join(options.root || app.root, 'app', 'resources');

    try {
      existsSync(app._resourceDir)
    }
    catch (err) {
      //
      // If an invalid path has been provided, don't attempt to load it
      //
      console.error('invalid resource path: ' + app._resourceDir);
      return;
    }

    var files = common.tryReaddirSync(app._resourceDir);

    if (files.length === 0) {
      //
      // If no resources were found in the path, warn, but don't error
      //
      console.warn('no resources found at: ' + app._resourceDir);
    }

    files.forEach(function (file) {
      file = file.replace('.js', '');
      app.resources.__defineGetter__(common.capitalize(file), function () {
        delete app.resources[common.capitalize(file)];
        return app.resources[common.capitalize(file)] = require(
          path.resolve(app._resourceDir, file)
        );
      });
    });

  }
  //
  // TODO: Determine how best to integrate `restful` here.
  //
};

exports.init = function (done) {
  var app = this,
      options;

  //
  // Attempt to merge defaults passed to `app.use(flatiron.plugins.resourceful)`
  // with any additional configuration that may have been loaded.
  //
  options = common.mixin(
    {},
    app.options['resourceful'],
    app.config.get('resourceful') || app.config.get('database') || {}
  );

  app.config.set('resourceful', options);

  //
  // Remark: Should we accept the autoMigrate option?
  //
  if (options.engine) {
    resourceful.use(options.engine, options);
  }

  done();
};
