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

var cordova_util = require('./util'),
    ConfigParser = require('cordova-common').ConfigParser,
    Q            = require('q'),
    semver       = require('semver');


/** Returns all the platforms that are currently saved into config.xml
 *  @return {Promise<{name: string, version: string, src: string}[]>}
 *      e.g: [ {name: 'android', version: '3.5.0'}, {name: 'wp8', src: 'C:/path/to/platform'}, {name: 'ios', src: 'git://...'} ]
 */
function getPlatforms(projectRoot){
    var xml = cordova_util.projectConfig(projectRoot);
    var cfg = new ConfigParser(xml);

    // If an engine's 'version' property is really its source, map that to the appropriate field.
    var engines = cfg.getEngines().map(function (engine) {
        var result = {
            name: engine.name
        };

        if (semver.validRange(engine.spec, true)) {
            result.version = engine.spec;
        } else {
            result.src = engine.spec;
        }

        return result;
    });

    return Q(engines);
}

/** Returns all the plugins that are currently saved into config.xml
 *  @return {Promise<{id: string, version: string, variables: {name: string, value: string}[]}[]>}
 *      e.g: [ {id: 'org.apache.cordova.device', variables: [{name: 'APP_ID', value: 'my-app-id'}, {name: 'APP_NAME', value: 'my-app-name'}]} ]
 */
function getPlugins(projectRoot){
    var xml = cordova_util.projectConfig(projectRoot);
    var cfg = new ConfigParser(xml);

    // Map variables object to an array
    var plugins = cfg.getPlugins().map(function (plugin) {
        var result = {
            name: plugin.name
        };

        if (semver.validRange(plugin.spec, true)) {
            result.version = plugin.spec;
        } else {
            result.src = plugin.spec;
        }

        var variablesObject = plugin.variables;
        var variablesArray = [];
        if (variablesObject) {
            for (var variable in variablesObject) {
                variablesArray.push({
                    name: variable,
                    value: variablesObject[variable]
                });
            }
        }
        result.variables = variablesArray;
        return result;
    });

    return Q(plugins);
}

module.exports = {
    getPlatforms: getPlatforms,
    getPlugins: getPlugins
};
