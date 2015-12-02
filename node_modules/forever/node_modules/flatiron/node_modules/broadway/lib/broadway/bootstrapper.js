/*
 * bootstrapper.js: Default logic for bootstrapping broadway applications.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var broadway = require('../broadway');

//
// ### bootstrap (app, callback)
// #### @app {broadway.App} Application to bootstrap
// #### @callback {function} Continuation to respond to when complete.
// Bootstraps the specified `app`.
//
exports.bootstrap = function (app) {
  app.options['config']        = app.options['config'] || {};
  app.options['config'].init   = false;
  app.use(broadway.plugins.config);

  //
  // Remove initializers run by the bootstrapper.
  //
  delete app.initializers['config'];
  app.initlist.pop();

  //
  // Set the current environment in the config
  //
  app.config.set('env', app.env);
};

//
// ### bootstrap (app, callback)
// #### @app {broadway.App} Application to bootstrap
// #### @callback {function} Continuation to respond to when complete.
// Runs the initialization step of the bootstrapping process
// for the specified `app`.
//
exports.init = function (app, callback) {
  broadway.plugins.config.init.call(app, function (err) {
    if (err) {
      return callback(err);
    }

    if (app.config.get('handleExceptions')) {
      app.use(broadway.plugins.exceptions, app.options['exceptions'] || {});
    }

    app.use(broadway.plugins.directories, app.options['directories'] || {});
    app.use(broadway.plugins.log, app.options['log'] || {});

    //
    // Ensure the `directories` and `log` plugins initialize before
    // any other plugins. Since we cannot depend on ordering (if they were
    // manually added) splice the specific indexes
    //
    var log = app.initlist.indexOf('log');
    app.initlist.unshift.apply(
      app.initlist,
      app.initlist.splice(log)
    );

    var directories = app.initlist.indexOf('directories');
    app.initlist.unshift.apply(
      app.initlist,
      app.initlist.splice(directories)
    );

    //
    // Put the godot plugin before the log if it exists
    //
    var godot = app.initlist.indexOf('godot');
    if(~godot) {
      app.initlist.unshift.apply(
        app.initlist,
        app.initlist.splice(godot)
      );
    }

    callback();
  });
};
