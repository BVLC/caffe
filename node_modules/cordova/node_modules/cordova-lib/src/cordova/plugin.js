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

var cordova_util  = require('./util'),
    path          = require('path'),
    semver        = require('semver'),
    config        = require('./config'),
    Q             = require('q'),
    CordovaError  = require('cordova-common').CordovaError,
    ConfigParser  = require('cordova-common').ConfigParser,
    fs            = require('fs'),
    shell         = require('shelljs'),
    PluginInfoProvider = require('cordova-common').PluginInfoProvider,
    plugman       = require('../plugman/plugman'),
    pluginMapper  = require('cordova-registry-mapper').newToOld,
    events        = require('cordova-common').events,
    metadata      = require('../plugman/util/metadata'),
    chainMap      = require('../util/promise-util').Q_chainmap;

// Returns a promise.
module.exports = function plugin(command, targets, opts) {
    var projectRoot = cordova_util.cdProjectRoot();

    // Dance with all the possible call signatures we've come up over the time. They can be:
    // 1. plugin() -> list the plugins
    // 2. plugin(command, Array of targets, maybe opts object)
    // 3. plugin(command, target1, target2, target3 ... )
    // The targets are not really targets, they can be a mixture of plugins and options to be passed to plugman.

    command = command || 'ls';
    targets = targets || [];
    opts = opts || {};
    if ( opts.length ) {
        // This is the case with multiple targets as separate arguments and opts is not opts but another target.
        targets = Array.prototype.slice.call(arguments, 1);
        opts = {};
    }
    if ( !Array.isArray(targets) ) {
        // This means we had a single target given as string.
        targets = [targets];
    }
    opts.options = opts.options || [];
    opts.plugins = [];

    // TODO: Otherwise HooksRunner will be Object instead of function when run from tests - investigate why
    var HooksRunner = require('../hooks/HooksRunner');
    var hooksRunner = new HooksRunner(projectRoot);
    var config_json = config.read(projectRoot);
    var platformList = cordova_util.listPlatforms(projectRoot);

    // Massage plugin name(s) / path(s)
    var pluginPath = path.join(projectRoot, 'plugins');
    var plugins = cordova_util.findPlugins(pluginPath);
    if (!targets || !targets.length) {
        if (command == 'add' || command == 'rm') {
            return Q.reject(new CordovaError('You need to qualify `'+cordova_util.binname+' plugin add` or `'+cordova_util.binname+' plugin remove` with one or more plugins!'));
        } else {
            targets = [];
        }
    }

    //Split targets between plugins and options
    //Assume everything after a token with a '-' is an option
    var i;
    for (i = 0; i < targets.length; i++) {
        if (targets[i].match(/^-/)) {
            opts.options = targets.slice(i);
            break;
        } else {
            opts.plugins.push(targets[i]);
        }
    }

    switch(command) {
        case 'add':
            if (!targets || !targets.length) {
                return Q.reject(new CordovaError('No plugin specified. Please specify a plugin to add. See `'+cordova_util.binname+' plugin search`.'));
            }

            var xml = cordova_util.projectConfig(projectRoot);
            var cfg = new ConfigParser(xml);
            var searchPath = config_json.plugin_search_path || [];
            if (typeof opts.searchpath == 'string') {
                searchPath = opts.searchpath.split(path.delimiter).concat(searchPath);
            } else if (opts.searchpath) {
                searchPath = opts.searchpath.concat(searchPath);
            }
            // Blank it out to appease unit tests.
            if (searchPath.length === 0) {
                searchPath = undefined;
            }

            opts.cordova = { plugins: cordova_util.findPlugins(pluginPath) };
            return hooksRunner.fire('before_plugin_add', opts)
            .then(function() {
                var pluginInfoProvider = new PluginInfoProvider();
                return opts.plugins.reduce(function(soFar, target) {
                    return soFar.then(function() {
                        if (target[target.length - 1] == path.sep) {
                            target = target.substring(0, target.length - 1);
                        }

                        var parts = target.split('@');
                        var id = parts[0];
                        var version = parts[1];

                        // If no version is specified, retrieve the version (or source) from config.xml
                        if (!version && !cordova_util.isUrl(id) && !cordova_util.isDirectory(id)) {
                            events.emit('verbose', 'no version specified, retrieving version from config.xml');
                            var ver = getVersionFromConfigFile(id, cfg);

                            if (cordova_util.isUrl(ver) || cordova_util.isDirectory(ver)) {
                                target = ver;
                            } else {
                                target = ver ? (id + '@' + ver) : target;
                            }
                        }

                        // Fetch the plugin first.
                        events.emit('verbose', 'Calling plugman.fetch on plugin "' + target + '"');

                        var fetchOptions = {
                            searchpath: searchPath,
                            noregistry: opts.noregistry,
                            link: opts.link,
                            pluginInfoProvider: pluginInfoProvider,
                            variables: opts.cli_variables,
                            is_top_level: true
                        };

                        return plugman.raw.fetch(target, pluginPath, fetchOptions)
                        .then(function (directory) {
                            return pluginInfoProvider.get(directory);
                        });
                    })
                    .then(function(pluginInfo) {
                        // Validate top-level required variables
                        var pluginVariables = pluginInfo.getPreferences();
                        var missingVariables = Object.keys(pluginVariables)
                        .filter(function (variableName) {
                            // discard variables with default value
                            return !(pluginVariables[variableName] || opts.cli_variables[variableName]);
                        });

                        if (missingVariables.length) {
                            shell.rm('-rf', pluginInfo.dir);
                            var msg = 'Variable(s) missing (use: --variable ' + missingVariables.join('=value --variable ') + '=value).';
                            return Q.reject(new CordovaError(msg));
                        }

                        // Iterate (in serial!) over all platforms in the project and install the plugin.
                        return chainMap(platformList, function (platform) {
                            var platformRoot = path.join(projectRoot, 'platforms', platform),
                            options = {
                                cli_variables: opts.cli_variables || {},
                                browserify: opts.browserify || false,
                                searchpath: searchPath,
                                noregistry: opts.noregistry,
                                link: opts.link,
                                pluginInfoProvider: pluginInfoProvider
                            };

                            events.emit('verbose', 'Calling plugman.install on plugin "' + pluginInfo.dir + '" for platform "' + platform);
                            return plugman.raw.install(platform, platformRoot, path.basename(pluginInfo.dir), pluginPath, options);
                        })
                        .thenResolve(pluginInfo);
                    })
                    .then(function(pluginInfo){
                        // save to config.xml
                        if(saveToConfigXmlOn(config_json, opts)){
                            var src = parseSource(target, opts);
                            var attributes = {
                                name: pluginInfo.id,
                                spec: src ? src : '~' + pluginInfo.version
                            };

                            cfg.removePlugin(pluginInfo.id);
                            cfg.addPlugin(attributes, opts.cli_variables);
                            cfg.write();

                            events.emit('results', 'Saved plugin info for "' + pluginInfo.id + '" to config.xml');
                        }
                    });
                }, Q()); // end Q.all
            }).then(function() {
                // Need to require right here instead of doing this at the beginning of file
                // otherwise tests are failing without any real reason.
                return require('./prepare').preparePlatforms(platformList, projectRoot, opts);
            }).then(function() {
                opts.cordova = { plugins: cordova_util.findPlugins(pluginPath) };
                return hooksRunner.fire('after_plugin_add', opts);
            });
        case 'rm':
        case 'remove':
            if (!targets || !targets.length) {
                return Q.reject(new CordovaError('No plugin specified. Please specify a plugin to remove. See `'+cordova_util.binname+' plugin list`.'));
            }

            opts.cordova = { plugins: cordova_util.findPlugins(pluginPath) };
            return hooksRunner.fire('before_plugin_rm', opts)
            .then(function() {
                return opts.plugins.reduce(function(soFar, target) {
                    var validatedPluginId = validatePluginId(target, plugins);
                    if (!validatedPluginId) {
                        return Q.reject(new CordovaError('Plugin "' + target + '" is not present in the project. See `' + cordova_util.binname + ' plugin list`.'));
                    }
                    target = validatedPluginId;

                    // Iterate over all installed platforms and uninstall.
                    // If this is a web-only or dependency-only plugin, then
                    // there may be nothing to do here except remove the
                    // reference from the platform's plugin config JSON.
                    return platformList.reduce(function(soFar, platform) {
                        return soFar.then(function() {
                            var platformRoot = path.join(projectRoot, 'platforms', platform);
                            events.emit('verbose', 'Calling plugman.uninstall on plugin "' + target + '" for platform "' + platform + '"');
                            return plugman.raw.uninstall.uninstallPlatform(platform, platformRoot, target, pluginPath);
                        });
                    }, Q())
                    .then(function() {
                        // TODO: Should only uninstallPlugin when no platforms have it.
                        return plugman.raw.uninstall.uninstallPlugin(target, pluginPath);
                    }).then(function(){
                        //remove plugin from config.xml
                        if(saveToConfigXmlOn(config_json, opts)){
                            var configPath = cordova_util.projectConfig(projectRoot);
                            if(fs.existsSync(configPath)){//should not happen with real life but needed for tests
                                var configXml = new ConfigParser(configPath);
                                configXml.removePlugin(target);
                                configXml.write();
                                events.emit('results', 'config.xml entry for ' +target+ ' is removed');
                            }
                        }
                    })
                    .then(function(){
                        // Remove plugin from fetch.json
                        events.emit('verbose', 'Removing plugin ' + target + ' from fetch.json');
                        metadata.remove_fetch_metadata(pluginPath, target);
                    });
                }, Q());
            }).then(function () {
                return require('./prepare').preparePlatforms(platformList, projectRoot, opts);
            }).then(function() {
                opts.cordova = { plugins: cordova_util.findPlugins(pluginPath) };
                return hooksRunner.fire('after_plugin_rm', opts);
            });
        case 'search':
            return hooksRunner.fire('before_plugin_search')
            .then(function() {
                return plugman.raw.search(opts.plugins);
            }).then(function(plugins) {
                for(var plugin in plugins) {
                    events.emit('results', plugins[plugin].name, '-', plugins[plugin].description || 'no description provided');
                }
            }).then(function() {
                return hooksRunner.fire('after_plugin_search');
            });
        case 'save':
            // save the versions/folders/git-urls of currently installed plugins into config.xml
            return save(projectRoot, opts);
        default:
            return list(projectRoot, hooksRunner);
    }
};

function validatePluginId(pluginId, installedPlugins) {
    if (installedPlugins.indexOf(pluginId) >= 0) {
        return pluginId;
    }

    var oldStylePluginId = pluginMapper[pluginId];
    if (oldStylePluginId) {
        events.emit('log', 'Plugin "' + pluginId + '" is not present in the project. Converting value to "' + oldStylePluginId + '" and trying again.');
        return installedPlugins.indexOf(oldStylePluginId) >= 0 ? oldStylePluginId : null;
    }

    if (pluginId.indexOf('cordova-plugin-') < 0) {
        return validatePluginId('cordova-plugin-' + pluginId, installedPlugins);
    }
}

function save(projectRoot, opts){
    var xml = cordova_util.projectConfig(projectRoot);
    var cfg = new ConfigParser(xml);

    // First, remove all pre-existing plugins from config.xml
    cfg.getPluginIdList().forEach(function(plugin){
        cfg.removePlugin(plugin);
    });

    // Then, save top-level plugins and their sources
    var jsonFile = path.join(projectRoot, 'plugins', 'fetch.json');
    var plugins;
    try {
        // It might be the case that fetch.json file is not yet existent.
        // for example: when we have never ran the command 'cordova plugin add foo' on the project
        // in that case, there's nothing to do except bubble up the error
        plugins = JSON.parse(fs.readFileSync(jsonFile, 'utf-8'));
    } catch (err) {
        return Q.reject(err.message);
    }

    Object.keys(plugins).forEach(function(pluginName){
        var plugin = plugins[pluginName];
        var pluginSource = plugin.source;

        // If not a top-level plugin, skip it, don't save it to config.xml
        if(!plugin.is_top_level){
            return;
        }

        var attribs = {name: pluginName};
        var spec = getSpec(pluginSource, projectRoot, pluginName);
        if (spec) {
            attribs.spec = spec;
        }

        var variables = getPluginVariables(plugin.variables);
        cfg.addPlugin(attribs, variables);
    });
    cfg.write();

    return Q.resolve();
}

function getPluginVariables(variables){
    var result = [];
    if(!variables){
        return result;
    }

    Object.keys(variables).forEach(function(pluginVar){
        result.push({name: pluginVar, value: variables[pluginVar]});
    });

    return result;
}

function getVersionFromConfigFile(plugin, cfg){
    var pluginEntry = cfg.getPlugin(plugin);
    if (!pluginEntry) {
        // If the provided plugin id is in the new format (e.g. cordova-plugin-camera), it might be stored in config.xml
        // under the old format (e.g. org.apache.cordova.camera), so check for that.
        var oldStylePluginId = pluginMapper[plugin];
        if (oldStylePluginId) {
            pluginEntry = cfg.getPlugin(oldStylePluginId);
        }
    }
    return pluginEntry && pluginEntry.spec;
}

function list(projectRoot, hooksRunner) {
    var pluginsList = [];
    return hooksRunner.fire('before_plugin_ls')
    .then(function() {
        var pluginsDir = path.join(projectRoot, 'plugins');
        // TODO: This should list based off of platform.json, not directories within plugins/
        var pluginInfoProvider = new PluginInfoProvider();
        return pluginInfoProvider.getAllWithinSearchPath(pluginsDir);
    })
    .then(function(plugins) {
        if (plugins.length === 0) {
            events.emit('results', 'No plugins added. Use `'+cordova_util.binname+' plugin add <plugin>`.');
            return;
        }
        var pluginsDict = {};
        var lines = [];
        var txt, p;
        for (var i=0; i<plugins.length; i++) {
            p = plugins[i];
            pluginsDict[p.id] = p;
            pluginsList.push(p.id);
            txt = p.id + ' ' + p.version + ' "' + (p.name || p.description) + '"';
            lines.push(txt);
        }
        // Add warnings for deps with wrong versions.
        for (var id in pluginsDict) {
            p = pluginsDict[id];
            for (var depId in p.deps) {
                var dep = pluginsDict[depId];
                //events.emit('results', p.deps[depId].version);
                //events.emit('results', dep != null);
                if (!dep) {
                    txt = 'WARNING, missing dependency: plugin ' + id +
                          ' depends on ' + depId +
                          ' but it is not installed';
                    lines.push(txt);
                } else if (!semver.satisfies(dep.version, p.deps[depId].version)) {
                    txt = 'WARNING, broken dependency: plugin ' + id +
                          ' depends on ' + depId + ' ' + p.deps[depId].version +
                          ' but installed version is ' + dep.version;
                    lines.push(txt);
                }
            }
        }
        events.emit('results', lines.join('\n'));
    })
    .then(function() {
        return hooksRunner.fire('after_plugin_ls');
    })
    .then(function() {
        return pluginsList;
    });
}

function saveToConfigXmlOn(config_json, options){
    options = options || {};
    var autosave =  config_json.auto_save_plugins || false;
    return autosave || options.save;
}

function parseSource(target, opts) {
    var url = require('url');
    var uri = url.parse(target);
    if (uri.protocol && uri.protocol != 'file:' && uri.protocol[1] != ':' && !target.match(/^\w+:\\/)) {
        return target;
    } else {
        var plugin_dir = cordova_util.fixRelativePath(path.join(target, (opts.subdir || '.')));
        if (fs.existsSync(plugin_dir)) {
            return target;
        }
    }
    return null;
}

function getSpec(pluginSource, projectRoot, pluginName) {
    if (pluginSource.hasOwnProperty('url') || pluginSource.hasOwnProperty('path')) {
        return pluginSource.url || pluginSource.path;
    }

    var version = null;
    if (pluginSource.hasOwnProperty('id')) {
        // Note that currently version is only saved here if it was explicitly specified when the plugin was added.
        var parts = pluginSource.id.split('@');
        version = parts[1];
        if (version) {
            version = versionString(version);
        }
    }

    if (!version) {
        // Fallback on getting version from the plugin folder, if it's there
        var pluginInfoProvider = new PluginInfoProvider();
        var dir = path.join(projectRoot, 'plugins', pluginName);

        try {
            // pluginInfoProvider.get() will throw if directory does not exist.
            var pluginInfo = pluginInfoProvider.get(dir);
            if (pluginInfo) {
                version = versionString(pluginInfo.version);
            }
        } catch (err) {
        }
    }

    return version;
}

function versionString(version) {
    var validVersion = semver.valid(version, true);
    if (validVersion) {
        return '~' + validVersion;
    }

    if (semver.validRange(version, true)) {
        // Return what we were passed rather than the result of the validRange() call, as that call makes modifications
        // we don't want, like converting '^1.2.3' to '>=1.2.3-0 <2.0.0-0'
        return version;
    }

    return null;
}
