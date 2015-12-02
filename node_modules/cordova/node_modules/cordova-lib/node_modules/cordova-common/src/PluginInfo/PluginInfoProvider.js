/**
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
*/

/* jshint sub:true, laxcomma:true, laxbreak:true */

var fs = require('fs');
var path = require('path');
var PluginInfo = require('./PluginInfo');
var events = require('../events');

function PluginInfoProvider() {
    this._cache = {};
    this._getAllCache = {};
}

PluginInfoProvider.prototype.get = function(dirName) {
    var absPath = path.resolve(dirName);
    if (!this._cache[absPath]) {
        this._cache[absPath] = new PluginInfo(dirName);
    }
    return this._cache[absPath];
};

// Normally you don't need to put() entries, but it's used
// when copying plugins, and in unit tests.
PluginInfoProvider.prototype.put = function(pluginInfo) {
    var absPath = path.resolve(pluginInfo.dir);
    this._cache[absPath] = pluginInfo;
};

// Used for plugin search path processing.
// Given a dir containing multiple plugins, create a PluginInfo object for
// each of them and return as array.
// Should load them all in parallel and return a promise, but not yet.
PluginInfoProvider.prototype.getAllWithinSearchPath = function(dirName) {
    var absPath = path.resolve(dirName);
    if (!this._getAllCache[absPath]) {
        this._getAllCache[absPath] = getAllHelper(absPath, this);
    }
    return this._getAllCache[absPath];
};

function getAllHelper(absPath, provider) {
    if (!fs.existsSync(absPath)){
        return [];
    }
    // If dir itself is a plugin, return it in an array with one element.
    if (fs.existsSync(path.join(absPath, 'plugin.xml'))) {
        return [provider.get(absPath)];
    }
    var subdirs = fs.readdirSync(absPath);
    var plugins = [];
    subdirs.forEach(function(subdir) {
        var d = path.join(absPath, subdir);
        if (fs.existsSync(path.join(d, 'plugin.xml'))) {
            try {
                plugins.push(provider.get(d));
            } catch (e) {
                events.emit('warn', 'Error parsing ' + path.join(d, 'plugin.xml.\n' + e.stack));
            }
        }
    });
    return plugins;
}

module.exports = PluginInfoProvider;
