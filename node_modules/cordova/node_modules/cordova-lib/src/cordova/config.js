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

var path          = require('path'),
    fs            = require('fs'),
    url           = require('url'),
    shell         = require('shelljs');

// Map of project_root -> JSON
var configCache = {};
var autoPersist = true;

function config(project_root, opts) {
    var json = config.read(project_root);
    for (var p in opts) {
        json[p] = opts[p];
    }
    if (autoPersist) {
        config.write(project_root, json);
    } else {
        configCache[project_root] = JSON.stringify(json);
    }
    return json;
}

config.getAutoPersist = function() {
    return autoPersist;
};

config.setAutoPersist = function(value) {
    autoPersist = value;
};

config.read = function get_config(project_root) {
    var data = configCache[project_root];
    if (data === undefined) {
        var configPath = path.join(project_root, '.cordova', 'config.json');
        if (!fs.existsSync(configPath)) {
            data = '{}';
        } else {
            data = fs.readFileSync(configPath, 'utf-8');
        }
    }
    configCache[project_root] = data;
    return JSON.parse(data);
};

config.write = function set_config(project_root, json) {
    var configPath = path.join(project_root, '.cordova', 'config.json');
    var contents = JSON.stringify(json, null, 4);
    configCache[project_root] = contents;
    // Don't write the file for an empty config.
    if (contents != '{}' || fs.existsSync(configPath)) {
        shell.mkdir('-p', path.join(project_root, '.cordova'));
        fs.writeFileSync(configPath, contents, 'utf-8');
    }
    return json;
};

config.has_custom_path = function(project_root, platform) {
    var json = config.read(project_root);
    if (json.lib && json.lib[platform]) {
        var uri = url.parse(json.lib[platform].url || json.lib[platform].uri);
        if (!(uri.protocol)) return uri.path;
        else if (uri.protocol && uri.protocol[1] ==':') return uri.href;
    }
    return false;
};

module.exports = config;
