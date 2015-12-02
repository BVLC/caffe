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

var cordova_util      = require('./util'),
    ConfigParser      = require('cordova-common').ConfigParser,
    PlatformJson      = require('cordova-common').PlatformJson,
    PluginInfoProvider = require('cordova-common').PluginInfoProvider,
    events            = require('cordova-common').events,
    platforms         = require('../platforms/platforms'),
    HooksRunner       = require('../hooks/HooksRunner'),
    Q                 = require('q'),
    restore           = require('./restore-util'),
    path              = require('path'),
    browserify = require('../plugman/browserify'),
    config            = require('./config');

// Returns a promise.
exports = module.exports = prepare;
function prepare(options) {
    var projectRoot = cordova_util.cdProjectRoot();
    var config_json = config.read(projectRoot);
    options = options || { verbose: false, platforms: [], options: {} };

    var hooksRunner = new HooksRunner(projectRoot);
    return hooksRunner.fire('before_prepare', options)
    .then(function(){
        return restore.installPlatformsFromConfigXML(options.platforms, { searchpath : options.searchpath });
    })
    .then(function(){
        options = cordova_util.preProcessOptions(options);
        var paths = options.platforms.map(function(p) {
            var platform_path = path.join(projectRoot, 'platforms', p);
            return platforms.getPlatformApi(p, platform_path).getPlatformInfo().locations.www;
        });
        options.paths = paths;
    }).then(function() {
        options = cordova_util.preProcessOptions(options);
        options.searchpath = options.searchpath || config_json.plugin_search_path;
        // Iterate over each added platform
        return preparePlatforms(options.platforms, projectRoot, options);
    }).then(function() {
        options.paths = options.platforms.map(function(platform) {
            return platforms.getPlatformApi(platform).getPlatformInfo().locations.www;
        });
        return hooksRunner.fire('after_prepare', options);
    }).then(function () {
        return restore.installPluginsFromConfigXML(options);
    });
}

/**
 * Calls `platformApi.prepare` for each platform in project
 *
 * @param   {string[]}  platformList  List of platforms, added to current project
 * @param   {string}    projectRoot   Project root directory
 *
 * @return  {Promise}
 */
function preparePlatforms (platformList, projectRoot, options) {
    return Q.all(platformList.map(function(platform) {
        // TODO: this need to be replaced by real projectInfo
        // instance for current project.
        var project = {
            root: projectRoot,
            projectConfig: new ConfigParser(cordova_util.projectConfig(projectRoot)),
            locations: {
                plugins: path.join(projectRoot, 'plugins'),
                www: cordova_util.projectWww(projectRoot)
            }
        };

        // CB-9987 We need to reinstall the plugins for the platform it they were added by cordova@<5.4.0
        return restoreMissingPluginsForPlatform(platform, projectRoot, options)
        .then(function (argument) {
            // platformApi prepare takes care of all functionality
            // which previously had been executed by cordova.prepare:
            //   - reset config.xml and then merge changes from project's one,
            //   - update www directory from project's one and merge assets from platform_www,
            //   - reapply config changes, made by plugins,
            //   - update platform's project
            // Please note that plugins' changes, such as installes js files, assets and
            // config changes is not being reinstalled on each prepare.
            var platformApi = platforms.getPlatformApi(platform);
            return platformApi.prepare(project)
            .then(function () {
                if (options.browserify)
                    return browserify(project, platformApi);
            });
        });
    }));
}

module.exports.preparePlatforms = preparePlatforms;

/**
 * Ensures that plugins, installed with previous versions of CLI (<5.4.0) are
 *   readded to platform correctly. Also triggers regeneration of
 *   cordova_plugins.js file.
 *
 * @param   {String}  platform     Platform name to check for installed plugins
 * @param   {String}  projectRoot  A current cordova project location
 * @param   {Object}  [options]    Options that will be passed to
 *   PlatformApi.pluginAdd/Remove. This object will be extended with plugin
 *   variables, used to install the plugin initially (picked from "old"
 *   plugins/<platform>.json)
 *
 * @return  {Promise}               Promise that'll be fulfilled if all the
 *   plugins reinstalled properly.
 */
function restoreMissingPluginsForPlatform(platform, projectRoot, options) {
    events.emit('verbose', 'Searching PlatformJson files for differences between project vs. platform installed plugins');

    // Flow:
    // 1. Compare <platform>.json file in <project>/plugins ("old") and platforms/<platform> ("new")
    // 2. If there is any differences - merge "old" one into "new"
    // 3. Reinstall plugins that are missing and was merged on previous step

    var oldPlatformJson = PlatformJson.load(path.join(projectRoot, 'plugins'), platform);
    var platformJson = PlatformJson.load(path.join(projectRoot, 'platforms', platform), platform);

    var missingPlugins = Object.keys(oldPlatformJson.root.installed_plugins)
        .concat(Object.keys(oldPlatformJson.root.dependent_plugins))
        .reduce(function (result, candidate) {
            if (!platformJson.isPluginInstalled(candidate))
                result.push({name: candidate,
                    // Note: isPluginInstalled is actually returns not a boolean,
                    // but object which corresponds to this particular plugin
                    variables: oldPlatformJson.isPluginInstalled(candidate)});

            return result;
        }, []);

    if (missingPlugins.length === 0) {
        events.emit('verbose', 'No differences found between project and ' +
            platform + ' platform. Continuing...');
        return Q.resolve();
    }

    var api = platforms.getPlatformApi(platform);
    var provider = new PluginInfoProvider();
    return missingPlugins.reduce(function (promise, plugin) {
        return promise.then(function () {
            var pluginOptions = options || {};
            pluginOptions.variables = plugin.variables;
            pluginOptions.usePlatformWww = true;

            events.emit('verbose', 'Reinstalling missing plugin ' + plugin.name + ' to ' + platform + ' platform');
            var pluginInfo = provider.get(path.join(projectRoot, 'plugins', plugin.name));
            return api.removePlugin(pluginInfo, pluginOptions)
            .then(function () {
                return api.addPlugin(pluginInfo, pluginOptions);
            });
        });
    }, Q());
}
