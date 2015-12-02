/*
 * config.js: Default configuration management plugin which attachs nconf to App instances
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var nconf = require('nconf');

//
// ### Name this plugin
//
exports.name = 'config';

//
// ### function attach (options)
// #### @options {Object} Options for this plugin
// Extends `this` (the application) with configuration functionality
// from `nconf`.
//
exports.attach = function (options) {
  options = options  || {};
  this.config = new nconf.Provider(options);

  //
  // Setup a default store
  //
  this.config.use('literal');
  this.config.stores.literal.readOnly = false;
};

//
// ### function init (done)
// #### @done {function} Continuation to respond to when complete.
// Initalizes the `nconf.Provider` associated with this instance.
//
exports.init = function (done) {
  //
  // Remark: There should be code here for automated remote
  // seeding and loading
  //
  this.config.load(function (err) {
    return err ? done(err) : done();
  });
};