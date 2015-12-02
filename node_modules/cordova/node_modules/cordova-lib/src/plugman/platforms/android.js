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

/* jshint laxcomma:true, sub:true */

var fs = require('fs');
var path = require('path')
   , common = require('./common')
   , events = require('cordova-common').events
   , xml_helpers = require('cordova-common').xmlHelpers
   , properties_parser = require('properties-parser')
   , android_project = require('../util/android-project')
   , CordovaError = require('cordova-common').CordovaError
   ;
var semver = require('semver');

var projectFileCache = {};

function getProjectSdkDir(project_dir) {
    var localProperties = properties_parser.createEditor(path.resolve(project_dir, 'local.properties'));
    return localProperties.get('sdk.dir');
}

function getCustomSubprojectRelativeDir(plugin_id, project_dir, src) {
    // All custom subprojects are prefixed with the last portion of the package id.
    // This is to avoid collisions when opening multiple projects in Eclipse that have subprojects with the same name.
    var prefix = package_suffix(project_dir);
    var subRelativeDir = path.join(plugin_id, prefix + '-' + path.basename(src));
    return subRelativeDir;
}

function package_suffix(project_dir) {
    var packageName = module.exports.package_name(project_dir);
    var lastDotIndex = packageName.lastIndexOf('.');
    return packageName.substring(lastDotIndex + 1);
}

module.exports = {
    www_dir:function(project_dir) {
        return path.join(project_dir, 'assets', 'www');
    },
    // reads the package name out of the Android Manifest file
    // @param string project_dir the absolute path to the directory containing the project
    // @return string the name of the package
    package_name:function(project_dir) {
        var mDoc = xml_helpers.parseElementtreeSync(path.join(project_dir, 'AndroidManifest.xml'));

        return mDoc._root.attrib['package'];
    },
    'source-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            if (!src) {
                throw new CordovaError('<source-file> element is missing "src" attribute for plugin: ' + plugin_id);
            }
            var targetDir = obj.targetDir;
            if (!targetDir) {
                throw new CordovaError('<source-file> element is missing "target-dir" attribute for plugin: ' + plugin_id);
            }
            var dest = path.join(targetDir, path.basename(src));

            common.copyNewFile(plugin_dir, src, project_dir, dest, options && options.link);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var dest = path.join(obj.targetDir, path.basename(obj.src));
            common.deleteJava(project_dir, dest);
        }
    },
    'header-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.install is not supported for android');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.uninstall is not supported for android');
        }
    },
    'lib-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var dest = path.join('libs', path.basename(src));
            common.copyFile(plugin_dir, src, project_dir, dest, !!(options && options.link));
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var src = obj.src;
            var dest = path.join('libs', path.basename(src));
            common.removeFile(project_dir, dest);
        }
    },
    'resource-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var target = obj.target;
            events.emit('verbose', 'Copying resource file ' + src + ' to ' + target);
            common.copyFile(plugin_dir, src, project_dir, path.normalize(target), !!(options && options.link));
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var target = obj.target;
            common.removeFile(project_dir, path.normalize(target));
        }
    },
    'framework': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var custom = obj.custom;
            if (!src) throw new CordovaError('src not specified in <framework> for plugin: ' + plugin_id);

            events.emit('verbose', 'Installing Android library: ' + src);
            var parent = obj.parent;
            var parentDir = parent ? path.resolve(project_dir, parent) : project_dir;
            var subDir;
            var type = obj.type;

            if (custom) {
                var subRelativeDir = getCustomSubprojectRelativeDir(plugin_id, project_dir, src);
                common.copyNewFile(plugin_dir, src, project_dir, subRelativeDir, options && options.link);
                subDir = path.resolve(project_dir, subRelativeDir);
            } else {
                if (semver.gte(options.platformVersion, '4.0.0-dev')) {
                    type = 'sys';
                    subDir = src;
                } else {
                    var sdk_dir = getProjectSdkDir(project_dir);
                    subDir = path.resolve(sdk_dir, src);
                }
            }

            var projectConfig = module.exports.parseProjectFile(project_dir);
            if (type == 'gradleReference') {
                projectConfig.addGradleReference(parentDir, subDir);
            } else if (type == 'sys') {
                projectConfig.addSystemLibrary(parentDir, subDir);
            } else {
                projectConfig.addSubProject(parentDir, subDir);
                projectConfig.needsSubLibraryUpdate = semver.lt(options.platformVersion, '3.6.0');
            }
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var src = obj.src;
            var custom = obj.custom;
            if (!src) throw new CordovaError('src not specified in <framework> for plugin: ' + plugin_id);

            events.emit('verbose', 'Uninstalling Android library: ' + src);
            var parent = obj.parent;
            var parentDir = parent ? path.resolve(project_dir, parent) : project_dir;
            var subDir;
            var type = obj.type;

            if (custom) {
                var subRelativeDir = getCustomSubprojectRelativeDir(plugin_id, project_dir, src);
                common.removeFile(project_dir, subRelativeDir);
                subDir = path.resolve(project_dir, subRelativeDir);
                // If it's the last framework in the plugin, remove the parent directory.
                var parDir = path.dirname(subDir);
                if (fs.readdirSync(parDir).length === 0) {
                    fs.rmdirSync(parDir);
                }
            } else {
                if (semver.gte(options.platformVersion, '4.0.0-dev')) {
                    type = 'sys';
                    subDir = src;
                } else {
                    var sdk_dir = getProjectSdkDir(project_dir);
                    subDir = path.resolve(sdk_dir, src);
                }
            }

            var projectConfig = module.exports.parseProjectFile(project_dir);
            if (type == 'gradleReference') {
                projectConfig.removeGradleReference(parentDir, subDir);
            } else if (type == 'sys') {
                projectConfig.removeSystemLibrary(parentDir, subDir);
            } else {
                projectConfig.removeSubProject(parentDir, subDir);
            }
        }
    },
    parseProjectFile: function(project_dir){
        if (!projectFileCache[project_dir]) {
            projectFileCache[project_dir] = new android_project.AndroidProject(project_dir);
        }

        return projectFileCache[project_dir];
    },
    purgeProjectFileCache:function(project_dir) {
        delete projectFileCache[project_dir];
    }
};

