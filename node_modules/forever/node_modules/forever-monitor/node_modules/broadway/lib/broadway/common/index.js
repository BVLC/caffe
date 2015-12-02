/*
 * common.js: Top-level include for the `common` module.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var common = module.exports = require('utile');

common.directories = require('./directories');

// A naive shared "unique ID" generator for cases where `plugin.name` is
// undefined.
var id = 0;
common.uuid = function () {
  return String(id++);
}
