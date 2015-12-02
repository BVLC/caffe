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

var Q = require('q'),
    fs = require('fs'),
    path = require('path'),
    PluginInfo = require('cordova-common').PluginInfo,
    events = require('cordova-common').events,
    init = require('init-package-json');

//returns a promise
function createPackageJson(plugin_path) {
    var pluginInfo = new PluginInfo(plugin_path);

    var defaults = {
        id:pluginInfo.id,
        version:pluginInfo.version,
        description:pluginInfo.description,
        license:pluginInfo.license,
        keywords:pluginInfo.getKeywordsAndPlatforms(),
        repository:pluginInfo.repo,
        bugs:pluginInfo.issue,
        engines:pluginInfo.getEngines(),
        platforms: pluginInfo.getPlatformsArray()
    };

    fs.writeFile(path.join(__dirname,'defaults.json'), JSON.stringify(defaults), 'utf8', function (err) {
        if (err) throw err;
        events.emit('verbose', 'defaults.json created from plugin.xml');
        var initFile = require.resolve('./init-defaults');
        var dir = process.cwd();

        init(dir, initFile, {}, function (err, data) {
            if(err) throw err;
            events.emit('verbose', 'Package.json successfully created');
        });
    });
    return Q();
}

module.exports = createPackageJson;
