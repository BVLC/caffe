/*
 *
 * Copyright 2013 Anis Kadri
 *
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

/*
 * This module deals with shared configuration / dependency "stuff". That is:
 * - XML configuration files such as config.xml, AndroidManifest.xml or WMAppManifest.xml.
 * - plist files in iOS
 * Essentially, any type of shared resources that we need to handle with awareness
 * of how potentially multiple plugins depend on a single shared resource, should be
 * handled in this module.
 *
 * The implementation uses an object as a hash table, with "leaves" of the table tracking
 * reference counts.
 */

/* jshint sub:true */

var fs   = require('fs'),
    path = require('path'),
    et   = require('elementtree'),
    semver = require('semver'),
    events = require('../events'),
    ConfigKeeper = require('./ConfigKeeper');

var mungeutil = require('./munge-util');

exports.PlatformMunger = PlatformMunger;

exports.process = function(plugins_dir, project_dir, platform, platformJson, pluginInfoProvider) {
    var munger = new PlatformMunger(platform, project_dir, platformJson, pluginInfoProvider);
    munger.process(plugins_dir);
    munger.save_all();
};

/******************************************************************************
* PlatformMunger class
*
* Can deal with config file of a single project.
* Parsed config files are cached in a ConfigKeeper object.
******************************************************************************/
function PlatformMunger(platform, project_dir, platformJson, pluginInfoProvider) {
    this.platform = platform;
    this.project_dir = project_dir;
    this.config_keeper = new ConfigKeeper(project_dir);
    this.platformJson = platformJson;
    this.pluginInfoProvider = pluginInfoProvider;
}

// Write out all unsaved files.
PlatformMunger.prototype.save_all = PlatformMunger_save_all;
function PlatformMunger_save_all() {
    this.config_keeper.save_all();
    this.platformJson.save();
}

// Apply a munge object to a single config file.
// The remove parameter tells whether to add the change or remove it.
PlatformMunger.prototype.apply_file_munge = PlatformMunger_apply_file_munge;
function PlatformMunger_apply_file_munge(file, munge, remove) {
    var self = this;

    for (var selector in munge.parents) {
        for (var xml_child in munge.parents[selector]) {
            // this xml child is new, graft it (only if config file exists)
            var config_file = self.config_keeper.get(self.project_dir, self.platform, file);
            if (config_file.exists) {
                if (remove) config_file.prune_child(selector, munge.parents[selector][xml_child]);
                else config_file.graft_child(selector, munge.parents[selector][xml_child]);
            }
        }
    }
}


PlatformMunger.prototype.remove_plugin_changes = remove_plugin_changes;
function remove_plugin_changes(pluginInfo, is_top_level) {
    var self = this;
    var platform_config = self.platformJson.root;
    var plugin_vars = is_top_level ?
        platform_config.installed_plugins[pluginInfo.id] :
        platform_config.dependent_plugins[pluginInfo.id];

    // get config munge, aka how did this plugin change various config files
    var config_munge = self.generate_plugin_config_munge(pluginInfo, plugin_vars);
    // global munge looks at all plugins' changes to config files
    var global_munge = platform_config.config_munge;
    var munge = mungeutil.decrement_munge(global_munge, config_munge);

    for (var file in munge.files) {
        // CB-6976 Windows Universal Apps. Compatibility fix for existing plugins.
        if (self.platform == 'windows' && file == 'package.appxmanifest' &&
            !fs.existsSync(path.join(self.project_dir, 'package.appxmanifest'))) {
            // New windows template separate manifest files for Windows8, Windows8.1 and WP8.1
            var substs = ['package.phone.appxmanifest', 'package.windows.appxmanifest', 'package.windows80.appxmanifest', 'package.windows10.appxmanifest'];
            /* jshint loopfunc:true */
            substs.forEach(function(subst) {
                events.emit('verbose', 'Applying munge to ' + subst);
                self.apply_file_munge(subst, munge.files[file], true);
            });
            /* jshint loopfunc:false */
        }
        self.apply_file_munge(file, munge.files[file], /* remove = */ true);
    }

    // Remove from installed_plugins
    self.platformJson.removePlugin(pluginInfo.id, is_top_level);
    return self;
}


PlatformMunger.prototype.add_plugin_changes = add_plugin_changes;
function add_plugin_changes(pluginInfo, plugin_vars, is_top_level, should_increment) {
    var self = this;
    var platform_config = self.platformJson.root;

    // get config munge, aka how should this plugin change various config files
    var config_munge = self.generate_plugin_config_munge(pluginInfo, plugin_vars);
    // global munge looks at all plugins' changes to config files

    // TODO: The should_increment param is only used by cordova-cli and is going away soon.
    // If should_increment is set to false, avoid modifying the global_munge (use clone)
    // and apply the entire config_munge because it's already a proper subset of the global_munge.
    var munge, global_munge;
    if (should_increment) {
        global_munge = platform_config.config_munge;
        munge = mungeutil.increment_munge(global_munge, config_munge);
    } else {
        global_munge = mungeutil.clone_munge(platform_config.config_munge);
        munge = config_munge;
    }

    for (var file in munge.files) {
        // CB-6976 Windows Universal Apps. Compatibility fix for existing plugins.
        if (self.platform == 'windows' && file == 'package.appxmanifest' &&
            !fs.existsSync(path.join(self.project_dir, 'package.appxmanifest'))) {
            var substs = ['package.phone.appxmanifest', 'package.windows.appxmanifest', 'package.windows80.appxmanifest', 'package.windows10.appxmanifest'];
            /* jshint loopfunc:true */
            substs.forEach(function(subst) {
                events.emit('verbose', 'Applying munge to ' + subst);
                self.apply_file_munge(subst, munge.files[file]);
            });
            /* jshint loopfunc:false */
        }
        self.apply_file_munge(file, munge.files[file]);
    }

    // Move to installed/dependent_plugins
    self.platformJson.addPlugin(pluginInfo.id, plugin_vars || {}, is_top_level);
    return self;
}


// Load the global munge from platform json and apply all of it.
// Used by cordova prepare to re-generate some config file from platform
// defaults and the global munge.
PlatformMunger.prototype.reapply_global_munge = reapply_global_munge ;
function reapply_global_munge () {
    var self = this;

    var platform_config = self.platformJson.root;
    var global_munge = platform_config.config_munge;
    for (var file in global_munge.files) {
        self.apply_file_munge(file, global_munge.files[file]);
    }

    return self;
}


// generate_plugin_config_munge
// Generate the munge object from plugin.xml + vars
PlatformMunger.prototype.generate_plugin_config_munge = generate_plugin_config_munge;
function generate_plugin_config_munge(pluginInfo, vars) {
    var self = this;

    vars = vars || {};
    var munge = { files: {} };
    var changes = pluginInfo.getConfigFiles(self.platform);

    // Demux 'package.appxmanifest' into relevant platform-specific appx manifests.
    // Only spend the cycles if there are version-specific plugin settings
    if (self.platform === 'windows' &&
            changes.some(function(change) {
                return ((typeof change.versions !== 'undefined') ||
                    (typeof change.deviceTarget !== 'undefined'));
            }))
    {
        var manifests = {
            'windows': {
                '8.0.0': 'package.windows80.appxmanifest',
                '8.1.0': 'package.windows.appxmanifest',
                '10.0.0': 'package.windows10.appxmanifest'
            },
            'phone': {
                '8.1.0': 'package.phone.appxmanifest',
                '10.0.0': 'package.windows10.appxmanifest'
            },
            'all': {
                '8.0.0': 'package.windows80.appxmanifest',
                '8.1.0': ['package.windows.appxmanifest', 'package.phone.appxmanifest'],
                '10.0.0': 'package.windows10.appxmanifest'
            }
        };

        var oldChanges = changes;
        changes = [];

        oldChanges.forEach(function(change, changeIndex) {
            // Only support semver/device-target demux for package.appxmanifest
            // Pass through in case something downstream wants to use it
            if (change.target !== 'package.appxmanifest') {
                changes.push(change);
                return;
            }

            var hasVersion = (typeof change.versions !== 'undefined');
            var hasTargets = (typeof change.deviceTarget !== 'undefined');

            // No semver/device-target for this config-file, pass it through
            if (!(hasVersion || hasTargets)) {
                changes.push(change);
                return;
            }

            var targetDeviceSet = hasTargets ? change.deviceTarget : 'all';
            if (['windows', 'phone', 'all'].indexOf(targetDeviceSet) === -1) {
                // target-device couldn't be resolved, fix it up here to a valid value
                targetDeviceSet = 'all';
            }
            var knownWindowsVersionsForTargetDeviceSet = Object.keys(manifests[targetDeviceSet]);

            // at this point, 'change' targets package.appxmanifest and has a version attribute
            knownWindowsVersionsForTargetDeviceSet.forEach(function(winver) {
                // This is a local function that creates the new replacement representing the
                // mutation.  Used to save code further down.
                var createReplacement = function(manifestFile, originalChange) {
                    var replacement = {
                        target:         manifestFile,
                        parent:         originalChange.parent,
                        after:          originalChange.after,
                        xmls:           originalChange.xmls,
                        versions:       originalChange.versions,
                        deviceTarget:   originalChange.deviceTarget
                    };
                    return replacement;
                };

                // version doesn't satisfy, so skip
                if (hasVersion && !semver.satisfies(winver, change.versions)) {
                    return;
                }

                var versionSpecificManifests = manifests[targetDeviceSet][winver];
                if (versionSpecificManifests.constructor === Array) {
                    // e.g. all['8.1.0'] === ['pkg.windows.appxmanifest', 'pkg.phone.appxmanifest']
                    versionSpecificManifests.forEach(function(manifestFile) {
                        changes.push(createReplacement(manifestFile, change));
                    });
                }
                else {
                    // versionSpecificManifests is actually a single string
                    changes.push(createReplacement(versionSpecificManifests, change));
                }
            });
        });
    }

    changes.forEach(function(change) {
        change.xmls.forEach(function(xml) {
            // 1. stringify each xml
            var stringified = (new et.ElementTree(xml)).write({xml_declaration:false});
            // interp vars
            if (vars) {
                Object.keys(vars).forEach(function(key) {
                    var regExp = new RegExp('\\$' + key, 'g');
                    stringified = stringified.replace(regExp, vars[key]);
                });
            }
            // 2. add into munge
            mungeutil.deep_add(munge, change.target, change.parent, { xml: stringified, count: 1, after: change.after });
        });
    });
    return munge;
}

// Go over the prepare queue and apply the config munges for each plugin
// that has been (un)installed.
PlatformMunger.prototype.process = PlatformMunger_process;
function PlatformMunger_process(plugins_dir) {
    var self = this;
    var platform_config = self.platformJson.root;

    // Uninstallation first
    platform_config.prepare_queue.uninstalled.forEach(function(u) {
        var pluginInfo = self.pluginInfoProvider.get(path.join(plugins_dir, u.plugin));
        self.remove_plugin_changes(pluginInfo, u.topLevel);
    });

    // Now handle installation
    platform_config.prepare_queue.installed.forEach(function(u) {
        var pluginInfo = self.pluginInfoProvider.get(path.join(plugins_dir, u.plugin));
        self.add_plugin_changes(pluginInfo, u.vars, u.topLevel, true);
    });

    // Empty out installed/ uninstalled queues.
    platform_config.prepare_queue.uninstalled = [];
    platform_config.prepare_queue.installed = [];
}
/**** END of PlatformMunger ****/
