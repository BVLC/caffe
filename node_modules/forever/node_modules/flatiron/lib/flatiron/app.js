/*
 * app.js: Core Application object for managing plugins and features in broadway
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var fs = require('fs'),
    path = require('path'),
    util = require('util'),
    broadway = require('broadway');

var App = exports.App = function (options) {
  broadway.App.call(this, options);
};

//
// Inherit from `broadway.App`.
//
util.inherits(App, broadway.App);
