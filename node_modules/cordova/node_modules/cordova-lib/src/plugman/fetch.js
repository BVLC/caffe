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

var shell   = require('shelljs'),
    fs      = require('fs'),
    url     = require('url'),
    underscore = require('underscore'),
    semver = require('semver'),
    PluginInfoProvider = require('cordova-common').PluginInfoProvider,
    plugins = require('./util/plugins'),
    CordovaError = require('cordova-common').CordovaError,
    events = require('cordova-common').events,
    metadata = require('./util/metadata'),
    path    = require('path'),
    Q       = require('q'),
    registry = require('./registry/registry'),
    pluginMappernto = require('cordova-registry-mapper').newToOld,
    pluginMapperotn = require('cordova-registry-mapper').oldToNew;
var cordovaUtil = require('../cordova/util');

// Cache of PluginInfo objects for plugins in search path.
var localPlugins = null;

// possible options: link, subdir, git_ref, client, expected_id
// Returns a promise.
module.exports = fetchPlugin;
function fetchPlugin(plugin_src, plugins_dir, options) {
    // Ensure the containing directory exists.
    shell.mkdir('-p', plugins_dir);

    options = options || {};
    options.subdir = options.subdir || '.';
    options.searchpath = options.searchpath || [];
    if ( typeof options.searchpath === 'string' ) {
        options.searchpath = options.searchpath.split(path.delimiter);
    }

    var pluginInfoProvider = options.pluginInfoProvider || new PluginInfoProvider();

    // clone from git repository
    var uri = url.parse(plugin_src);

    // If the hash exists, it has the form from npm: http://foo.com/bar#git-ref[:subdir]
    // git-ref can be a commit SHA, a tag, or a branch
    // NB: No leading or trailing slash on the subdir.
    if (uri.hash) {
        var result = uri.hash.match(/^#([^:]*)(?::\/?(.*?)\/?)?$/);
        if (result) {
            if (result[1])
                options.git_ref = result[1];
            if (result[2])
                options.subdir = result[2];

            // Recurse and exit with the new options and truncated URL.
            var new_dir = plugin_src.substring(0, plugin_src.indexOf('#'));
            return fetchPlugin(new_dir, plugins_dir, options);
        }
    }

    return Q.when().then(function() {
        // If it looks like a network URL, git clone it.
        if ( uri.protocol && uri.protocol != 'file:' && uri.protocol[1] != ':' && !plugin_src.match(/^\w+:\\/)) {
            events.emit('log', 'Fetching plugin "' + plugin_src + '" via git clone');
            if (options.link) {
                events.emit('log', '--link is not supported for git URLs and will be ignored');
            }

            return plugins.clonePluginGit(plugin_src, plugins_dir, options)
            .fail(function (error) {
                var message = 'Failed to fetch plugin ' + plugin_src + ' via git.' +
                    '\nEither there is a connection problems, or plugin spec is incorrect:\n\t' + error;
                return Q.reject(new CordovaError(message));
            })
            .then(function(dir) {
                return {
                    pinfo: pluginInfoProvider.get(dir),
                    dest: dir,
                    fetchJsonSource: {
                        type: 'git',
                        url:  plugin_src,
                        subdir: options.subdir,
                        ref: options.git_ref
                    }
                };
            });
        }
        // If it's not a network URL, it's either a local path or a plugin ID.
        var plugin_dir = cordovaUtil.fixRelativePath(path.join(plugin_src, options.subdir));

        return Q.when().then(function() {
            if (fs.existsSync(plugin_dir)) {
                return {
                    pinfo: pluginInfoProvider.get(plugin_dir),
                    fetchJsonSource: {
                        type: 'local',
                        path: plugin_dir
                    }
                };
            }
            // If there is no such local path, it's a plugin id or id@versionspec.
            // First look for it in the local search path (if provided).
            var pinfo = findLocalPlugin(plugin_src, options.searchpath, pluginInfoProvider);
            if (pinfo) {
                events.emit('verbose', 'Found ' + plugin_src + ' at ' + pinfo.dir);
                return {
                    pinfo: pinfo,
                    fetchJsonSource: {
                        type: 'local',
                        path: pinfo.dir
                    }
                };
            } else if ( options.noregistry ) {
                return Q.reject(new CordovaError(
                    'Plugin ' + plugin_src + ' not found locally. ' +
                    'Note, plugin registry was disabled by --noregistry flag.'
                ));
            }
            // If not found in local search path, fetch from the registry.
            var newID = pluginMapperotn[plugin_src];
            if(newID) {
                events.emit('warn', 'Notice: ' + plugin_src + ' has been automatically converted to ' + newID + ' and fetched from npm. This is due to our old plugins registry shutting down.');                
                plugin_src = newID;
            } 
            return registry.fetch([plugin_src], options.client)
            .fail(function (error) {
                var message = 'Failed to fetch plugin ' + plugin_src + ' via registry.' +
                    '\nProbably this is either a connection problem, or plugin spec is incorrect.' +
                    '\nCheck your connection and plugin name/version/URL.' +
                    '\n' + error;
                return Q.reject(new CordovaError(message));
            })
            .then(function(dir) {
                return {
                        pinfo: pluginInfoProvider.get(dir),
                    fetchJsonSource: {
                        type: 'registry',
                        id: plugin_src
                    }
                };
            });
        }).then(function(result) {
            options.plugin_src_dir = result.pinfo.dir;
            return Q.when(copyPlugin(result.pinfo, plugins_dir, options.link && result.fetchJsonSource.type == 'local'))
            .then(function(dir) {
                result.dest = dir;
                return result;
            });
        });
    }).then(function(result){
        checkID(options.expected_id, result.pinfo);
        var data = { source: result.fetchJsonSource };
        data.is_top_level = options.is_top_level;
        data.variables = options.variables || {};
        metadata.save_fetch_metadata(plugins_dir, result.pinfo.id, data);
        return result.dest;
    });
}

// Helper function for checking expected plugin IDs against reality.
function checkID(expectedIdAndVersion, pinfo) {
    if (!expectedIdAndVersion) return;
    var expectedId = expectedIdAndVersion.split('@')[0];
    var expectedVersion = expectedIdAndVersion.split('@')[1];
    if (expectedId != pinfo.id) {
        throw new Error('Expected plugin to have ID "' + expectedId + '" but got "' + pinfo.id + '".');
    }
    if (expectedVersion && !semver.satisfies(pinfo.version, expectedVersion)) {
        throw new Error('Expected plugin ' + pinfo.id + ' to satisfy version "' + expectedVersion + '" but got "' + pinfo.version + '".');
    }
}

// Note, there is no cache invalidation logic for local plugins.
// As of this writing loadLocalPlugins() is never called with different
// search paths and such case would not be handled properly.
function loadLocalPlugins(searchpath, pluginInfoProvider) {
    if (localPlugins) {
        // localPlugins already populated, nothing to do.
        // just in case, make sure it was loaded with the same search path
        if ( !underscore.isEqual(localPlugins.searchpath, searchpath) ) {
            var msg =
                'loadLocalPlugins called twice with different search paths.' +
                'Support for this is not implemented.';
            throw new Error(msg);
        }
        return;
    }

    // Populate localPlugins object.
    localPlugins = {};
    localPlugins.searchpath = searchpath;
    localPlugins.plugins = {};

    searchpath.forEach(function(dir) {
        var ps = pluginInfoProvider.getAllWithinSearchPath(dir);
        ps.forEach(function(p) {
            var versions = localPlugins.plugins[p.id] || [];
            versions.push(p);
            localPlugins.plugins[p.id] = versions;
        });
    });
}


// If a plugin is fund in local search path, return a PluginInfo for it.
// Ignore plugins that don't satisfy the required version spec.
// If several versions are present in search path, return the latest.
// Examples of accepted plugin_src strings:
//      org.apache.cordova.file
//      org.apache.cordova.file@>=1.2.0
function findLocalPlugin(plugin_src, searchpath, pluginInfoProvider) {
    loadLocalPlugins(searchpath, pluginInfoProvider);
    var id = plugin_src;
    var versionspec = '*';
    if (plugin_src.indexOf('@') != -1) {
        var parts = plugin_src.split('@');
        id = parts[0];
        versionspec = parts[1];
    }

    var latest = null;
    var versions = localPlugins.plugins[id];

    if (!versions) return null;

    versions.forEach(function(pinfo) {
        // Ignore versions that don't satisfy the the requested version range.
        // Ignore -dev suffix because latest semver versions doesn't handle it properly (CB-9421)
        if (!semver.satisfies(pinfo.version.replace(/-dev$/, ''), versionspec)) {
            return;
        }
        if (!latest) {
            latest = pinfo;
            return;
        }
        if (semver.gt(pinfo.version, latest.version)) {
            latest = pinfo;
        }

    });
    return latest;
}


// Copy or link a plugin from plugin_dir to plugins_dir/plugin_id.
// if alternative ID of plugin exists in plugins_dir/plugin_id, skip copying
function copyPlugin(pinfo, plugins_dir, link) {

    var plugin_dir = pinfo.dir;
    var dest = path.join(plugins_dir, pinfo.id);
    var altDest;

    //check if alternative id already exists in plugins directory
    if(pluginMapperotn[pinfo.id]) {
        altDest = path.join(plugins_dir, pluginMapperotn[pinfo.id]);
    } else if(pluginMappernto[pinfo.id]) {
        altDest = path.join(plugins_dir, pluginMappernto[pinfo.id]);
    }

    if(fs.existsSync(altDest)) {
        events.emit('log', pinfo.id + '" will not install due to "' + altDest + '" being installed.');
        return altDest;
    }

    shell.rm('-rf', dest);
    if (link) {
        var isRelativePath = plugin_dir.charAt(1) != ':' && plugin_dir.charAt(0) != path.sep;
        var fixedPath = isRelativePath ? path.join(path.relative(plugins_dir, process.env.PWD || process.cwd()), plugin_dir) : plugin_dir;
        events.emit('verbose', 'Linking "' + dest + '" => "' + fixedPath + '"');
        fs.symlinkSync(fixedPath, dest, 'dir');
    } else {
        shell.mkdir('-p', dest);
        events.emit('verbose', 'Copying plugin "' + plugin_dir + '" => "' + dest + '"');
        shell.cp('-R', path.join(plugin_dir, '*') , dest);
    }
    return dest;
}
