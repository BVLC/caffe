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

var fs = require('fs');
var path = require('path');

var cachedJson = null;

function getJson(pluginsDir) {
    if (!cachedJson) {
        var fetchJsonPath = path.join(pluginsDir, 'fetch.json');
        if (fs.existsSync(fetchJsonPath)) {
            cachedJson = JSON.parse(fs.readFileSync(fetchJsonPath, 'utf-8'));
        } else {
            cachedJson = {};
        }
    }
    return cachedJson;
}

exports.get_fetch_metadata = function(plugin_dir) {
    var pluginsDir = path.dirname(plugin_dir);
    var pluginId = path.basename(plugin_dir);

    var metadataJson = getJson(pluginsDir);
    if (metadataJson[pluginId]) {
        return metadataJson[pluginId];
    }
    var legacyPath = path.join(plugin_dir, '.fetch.json');
    if (fs.existsSync(legacyPath)) {
        var ret = JSON.parse(fs.readFileSync(legacyPath, 'utf-8'));
        exports.save_fetch_metadata(pluginsDir, pluginId, ret);
        fs.unlinkSync(legacyPath);
        return ret;
    }
    return {};
};

exports.save_fetch_metadata = function(pluginsDir, pluginId, data) {
    var metadataJson = getJson(pluginsDir);
    metadataJson[pluginId] = data;
    var fetchJsonPath = path.join(pluginsDir, 'fetch.json');
    fs.writeFileSync(fetchJsonPath, JSON.stringify(metadataJson, null, 4), 'utf-8');
};

exports.remove_fetch_metadata = function(pluginsDir, pluginId){
    var metadataJson = getJson(pluginsDir);
    delete metadataJson[pluginId];
    var fetchJsonPath = path.join(pluginsDir, 'fetch.json');
    fs.writeFileSync(fetchJsonPath, JSON.stringify(metadataJson, null, 4), 'utf-8');
};

