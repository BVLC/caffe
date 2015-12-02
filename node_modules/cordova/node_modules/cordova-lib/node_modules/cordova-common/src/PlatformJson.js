/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
*/
/* jshint sub:true */

var fs = require('fs');
var path = require('path');
var shelljs = require('shelljs');
var mungeutil = require('./ConfigChanges/munge-util');
var pluginMappernto = require('cordova-registry-mapper').newToOld;
var pluginMapperotn = require('cordova-registry-mapper').oldToNew;

function PlatformJson(filePath, platform, root) {
    this.filePath = filePath;
    this.platform = platform;
    this.root = fix_munge(root || {});
}

PlatformJson.load = function(plugins_dir, platform) {
    var filePath = path.join(plugins_dir, platform + '.json');
    var root = null;
    if (fs.existsSync(filePath)) {
        root = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
    return new PlatformJson(filePath, platform, root);
};

PlatformJson.prototype.save = function() {
    shelljs.mkdir('-p', path.dirname(this.filePath));
    fs.writeFileSync(this.filePath, JSON.stringify(this.root, null, 4), 'utf-8');
};

/**
 * Indicates whether the specified plugin is installed as a top-level (not as
 *  dependency to others)
 * @method function
 * @param  {String} pluginId A plugin id to check for.
 * @return {Boolean} true if plugin installed as top-level, otherwise false.
 */
PlatformJson.prototype.isPluginTopLevel = function(pluginId) {
    var installedPlugins = this.root.installed_plugins;
    return installedPlugins[pluginId] ||
        installedPlugins[pluginMappernto[pluginId]] ||
        installedPlugins[pluginMapperotn[pluginId]];
};

/**
 * Indicates whether the specified plugin is installed as a dependency to other
 *  plugin.
 * @method function
 * @param  {String} pluginId A plugin id to check for.
 * @return {Boolean} true if plugin installed as a dependency, otherwise false.
 */
PlatformJson.prototype.isPluginDependent = function(pluginId) {
    var dependentPlugins = this.root.dependent_plugins;
    return dependentPlugins[pluginId] ||
        dependentPlugins[pluginMappernto[pluginId]] ||
        dependentPlugins[pluginMapperotn[pluginId]];
};

/**
 * Indicates whether plugin is installed either as top-level or as dependency.
 * @method function
 * @param  {String} pluginId A plugin id to check for.
 * @return {Boolean} true if plugin installed, otherwise false.
 */
PlatformJson.prototype.isPluginInstalled = function(pluginId) {
    return this.isPluginTopLevel(pluginId) ||
        this.isPluginDependent(pluginId);
};

PlatformJson.prototype.addPlugin = function(pluginId, variables, isTopLevel) {
    var pluginsList = isTopLevel ?
        this.root.installed_plugins :
        this.root.dependent_plugins;

    pluginsList[pluginId] = variables;

    return this;
};

PlatformJson.prototype.removePlugin = function(pluginId, isTopLevel) {
    var pluginsList = isTopLevel ?
        this.root.installed_plugins :
        this.root.dependent_plugins;

    delete pluginsList[pluginId];

    return this;
};

PlatformJson.prototype.addInstalledPluginToPrepareQueue = function(pluginDirName, vars, is_top_level) {
    this.root.prepare_queue.installed.push({'plugin':pluginDirName, 'vars':vars, 'topLevel':is_top_level});
};

PlatformJson.prototype.addUninstalledPluginToPrepareQueue = function(pluginId, is_top_level) {
    this.root.prepare_queue.uninstalled.push({'plugin':pluginId, 'id':pluginId, 'topLevel':is_top_level});
};

/**
 * Moves plugin, specified by id to top-level plugins. If plugin is top-level
 *  already, then does nothing.
 * @method function
 * @param  {String} pluginId A plugin id to make top-level.
 * @return {PlatformJson} PlatformJson instance.
 */
PlatformJson.prototype.makeTopLevel = function(pluginId) {
    var plugin = this.root.dependent_plugins[pluginId];
    if (plugin) {
        delete this.root.dependent_plugins[pluginId];
        this.root.installed_plugins[pluginId] = plugin;
    }
    return this;
};

// convert a munge from the old format ([file][parent][xml] = count) to the current one
function fix_munge(root) {
    root.prepare_queue = root.prepare_queue || {installed:[], uninstalled:[]};
    root.config_munge = root.config_munge || {files: {}};
    root.installed_plugins = root.installed_plugins || {};
    root.dependent_plugins = root.dependent_plugins || {};

    var munge = root.config_munge;
    if (!munge.files) {
        var new_munge = { files: {} };
        for (var file in munge) {
            for (var selector in munge[file]) {
                for (var xml_child in munge[file][selector]) {
                    var val = parseInt(munge[file][selector][xml_child]);
                    for (var i = 0; i < val; i++) {
                        mungeutil.deep_add(new_munge, [file, selector, { xml: xml_child, count: val }]);
                    }
                }
            }
        }
        root.config_munge = new_munge;
    }

    return root;
}

module.exports = PlatformJson;

