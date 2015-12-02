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

/* jshint expr:true */

var dep_graph = require('dep-graph'),
    path = require('path'),
    fs = require('fs'),
    underscore = require('underscore'),
    events = require('cordova-common').events,
    package;

module.exports = package = {

    generateDependencyInfo:function(platformJson, plugins_dir, pluginInfoProvider) {
        var json = platformJson.root;

        // TODO: store whole dependency tree in plugins/[platform].json
        // in case plugins are forcefully removed...
        var tlps = [];
        var graph = new dep_graph();
        Object.keys(json.installed_plugins).forEach(function(plugin_id) {
            tlps.push(plugin_id);

            var plugin_dir = path.join(plugins_dir, plugin_id);
            var pluginInfo = pluginInfoProvider.get(plugin_dir);
            var deps = pluginInfo.getDependencies(platformJson.platform);
            deps.forEach(function(dep) {
                graph.add(plugin_id, dep.id);
            });
        });
        Object.keys(json.dependent_plugins).forEach(function(plugin_id) {
            var plugin_dir = path.join(plugins_dir, plugin_id);
            // dependency plugin does not exist (CB-7846)
            if (!fs.existsSync(plugin_dir)) {
                events.emit('verbose', 'Plugin "'+ plugin_id +'" does not exist ('+ plugin_dir+')');
                return;
            }

            var pluginInfo = pluginInfoProvider.get(plugin_dir);
            var deps = pluginInfo.getDependencies(platformJson.platform);
            deps.forEach(function(dep) {
                graph.add(plugin_id, dep.id);
            });
        });

        return {
            graph:graph,
            top_level_plugins:tlps
        };
    },

    // Returns a list of top-level plugins which are (transitively) dependent on the given plugin.
    dependents: function(plugin_id, plugins_dir, platformJson, pluginInfoProvider) {
        var depsInfo;
        if(typeof plugins_dir == 'object')
            depsInfo = plugins_dir;
        else
            depsInfo = package.generateDependencyInfo(platformJson, plugins_dir, pluginInfoProvider);

        var graph = depsInfo.graph;
        var tlps = depsInfo.top_level_plugins;
        var dependents = tlps.filter(function(tlp) {
            return tlp != plugin_id && graph.getChain(tlp).indexOf(plugin_id) >= 0;
        });

        return dependents;
    },

    // Returns a list of plugins which the given plugin depends on, for which it is the only dependent.
    // In other words, if the given plugin were deleted, these dangling dependencies should be deleted too.
    danglers: function(plugin_id, plugins_dir, platformJson, pluginInfoProvider) {
        var depsInfo;
        if(typeof plugins_dir == 'object')
            depsInfo = plugins_dir;
        else
            depsInfo = package.generateDependencyInfo(platformJson, plugins_dir, pluginInfoProvider);

        var graph = depsInfo.graph;
        var dependencies = graph.getChain(plugin_id);

        var tlps = depsInfo.top_level_plugins;
        var diff_arr = [];
        tlps.forEach(function(tlp) {
            if (tlp != plugin_id) {
                diff_arr.push(graph.getChain(tlp));
            }
        });

        // if this plugin has dependencies, do a set difference to determine which dependencies are not required by other existing plugins
        diff_arr.unshift(dependencies);
        var danglers = underscore.difference.apply(null, diff_arr);

        // Ensure no top-level plugins are tagged as danglers.
        danglers = danglers && danglers.filter(function(x) { return tlps.indexOf(x) < 0; });
        return danglers;
    }
};
