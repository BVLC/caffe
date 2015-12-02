/*
 * common.js: Common utility functions for flatiron.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var fs = require('fs'),
    broadway = require('broadway');

//
// Hoist `broadway.common` to `flatiron.common`.
//
var common = module.exports = broadway.common.mixin({}, broadway.common);

//
// ### function templateUsage (app, commands)
// Updates the references to `<app>` to `app.name` in usage for the
// specified `commands`.
//
common.templateUsage = function (app, commands) {
  if (!app.name) {
    return commands;
  }

  function templateUsage(usage) {
    return usage.map(function (line) {
      return line.replace(/\<app\>/ig, app.name);
    });
  }

  Object.keys(commands).forEach(function (command) {
    if (command === 'usage') {
      commands.usage = templateUsage(commands.usage);
    }
    else if (commands[command].usage) {
      commands[command].usage = templateUsage(commands[command].usage);
    }
  });
};

//
// ### function tryReaddirSync (dir)
// #### @dir {string} Directory to attempt to list
//
// Attempts to call `fs.readdirSync` but ignores errors.
//
common.tryReaddirSync = function (dir) {
  try { return fs.readdirSync(dir) }
  catch (err) { return [] }
};