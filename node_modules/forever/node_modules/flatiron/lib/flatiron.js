/*
 * flatiron.js: An elegant blend of convention and configuration for building apps in Node.js and the browser.
 *
 * Copyright(c) 2011 Nodejitsu Inc. <info@nodejitsu.com>
 * MIT LICENCE
 *
 */

var fs = require('fs'),
    path = require('path'),
    broadway = require('broadway');

var flatiron = exports,
    _app;

//
// ### Export core `flatiron` modules
//
flatiron.common    = require('./flatiron/common');
flatiron.constants = require('./flatiron/constants');
flatiron.formats   = broadway.formats;
flatiron.App       = require('./flatiron/app').App;
flatiron.version   = require('../package.json').version;

//
// ### Expose core `flatiron` plugins
// Hoist those up from `broadway` and define each of
// the `flatiron` plugins as a lazy loaded `require` statement
//
flatiron.plugins = broadway.common.mixin(
  {},
  broadway.plugins,
  broadway.common.requireDirLazy(path.join(__dirname, 'flatiron', 'plugins'))
);


Object.defineProperty(flatiron, 'app', {

  // Don't allow another `.defineProperty` on 'app'
  configurable: false,

  //
  // ### getter @app {flatiron.App}
  // Gets the default top-level Application for `flatiron`
  //
  get: function() {
    return _app = _app || flatiron.createApp();
  },

  //
  // #### setter @app {flatiron.App}
  // Options for the application to create or the application to set
  //
  set: function(value) {
    if (value instanceof flatiron.App) return _app = value;
    return _app = flatiron.createApp(value);
  }

});


//
// ### function createApp (options)
// #### @options {Object} Options for the application to create
// Creates a new instance of `flatiron.App` with the
// specified `options`.
//
flatiron.createApp = function (options) {
  return new flatiron.App(options);
};

