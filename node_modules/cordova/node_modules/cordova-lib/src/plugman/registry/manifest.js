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

var path = require('path'),
    Q = require('q'),
    fs = require('fs'),
    whitelist = require('./whitelist');

var PluginInfo = require('cordova-common').PluginInfo;

function validateName(name) {
    if (!name.match(/^(\S+\.){2,}.*$/)) {
        throw new Error('Invalid plugin ID. It has to follow the reverse domain `com.domain.plugin` format');
    }

    if (name.match(/org.apache.cordova\..*/) && whitelist.indexOf(name) === -1) {
        throw new Error('Invalid Plugin ID. The "org.apache.cordova" prefix is reserved for plugins provided directly by the Cordova project.');
    }

    return true;
}

// Returns a promise.
function generatePackageJsonFromPluginXml(plugin_path) {
    return Q().then(function() {
        var package_json = {};
        var pluginInfo = new PluginInfo(plugin_path);

        // REQUIRED: name, version
        // OPTIONAL: description, license, keywords, engine
        var name = pluginInfo.id,
            version = pluginInfo.version,
            cordova_name = pluginInfo.name,
            description = pluginInfo.description,
            license = pluginInfo.license,
            keywords = pluginInfo.keywords,
            repo = pluginInfo.repo,
            issue = pluginInfo.issue,
            engines = pluginInfo.getEngines(),
            platforms = pluginInfo.getPlatforms(),
            englishdoc = '';

        if(!version) throw new Error('`version` required');

        package_json.version = version;

        if(!name) throw new Error('`id` is required');

        validateName(name);

        package_json.name = name.toLowerCase();

        if(cordova_name) package_json.cordova_name = cordova_name;
        if(description)  package_json.description  = description;
        if(license)      package_json.license      = license;
        if(repo)         package_json.repo         = repo;
        if(issue)        package_json.issue        = issue;
        if(keywords)     package_json.keywords     = keywords;
        if(platforms) {
            package_json.platforms = platforms.map(function(p) {
                return p.name;
            });
        }

        // Adding engines
        if(engines) {
            package_json.engines = engines.map(function(e) {
                return {name: e.name, version: e.version};
            });
        }

        // Set docs_path to doc/index.md exists
        var docs_path = path.resolve(plugin_path, 'doc/index.md');
        if(!(fs.existsSync(docs_path))){
            // Set docs_path to doc/en/index.md
            docs_path = path.resolve(plugin_path, 'doc/en/index.md');
        }
        if(fs.existsSync(docs_path)){
            englishdoc = fs.readFileSync(docs_path, 'utf-8');
            package_json.englishdoc = englishdoc;
        }

        // Write package.json
        var package_json_path = path.resolve(plugin_path, 'package.json');
        fs.writeFileSync(package_json_path, JSON.stringify(package_json, null, 4), 'utf8');
        return package_json;
    });
}

module.exports.generatePackageJsonFromPluginXml = generatePackageJsonFromPluginXml;
