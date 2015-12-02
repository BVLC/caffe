/*
 * broadway.js: Top-level include for the broadway module.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var path = require('path'),
    utile = require('utile');

var broadway = exports;

broadway.App      = require('./broadway/app').App;
broadway.common   = require('./broadway/common');
broadway.features = require('./broadway/features');
broadway.formats  = require('nconf').formats;
broadway.plugins  = utile.requireDirLazy(path.join(__dirname, 'broadway', 'plugins'));

