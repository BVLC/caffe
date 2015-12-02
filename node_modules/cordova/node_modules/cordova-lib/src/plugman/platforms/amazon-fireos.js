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

var path = require('path')
   , common = require('./common')
   , events = require('cordova-common').events
   , xml_helpers = require('cordova-common').xmlHelpers
   , properties_parser = require('properties-parser')
   , android_project = require('../util/android-project')
   , CordovaError = require('cordova-common').CordovaError
   ;

var projectFileCache = {};

module.exports = {
    www_dir:function(project_dir) {
        return path.join(project_dir, 'assets', 'www');
    },
    // reads the package name out of the Android Manifest file
    // @param string project_dir the absolute path to the directory containing the project
    // @return string the name of the package
    package_name:function (project_dir) {
        var mDoc = xml_helpers.parseElementtreeSync(path.join(project_dir, 'AndroidManifest.xml'));

        return mDoc._root.attrib['package'];
    },
    'source-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id) {
            var src = obj.src;
            if (!src) {
                throw new CordovaError('<source-file> element is missing "src" attribute for plugin: ' + plugin_id);
            }
            var targetDir = obj.targetDir;
            if (!targetDir) {
                throw new CordovaError('<source-file> element is missing "target-dir" attribute for plugin: ' + plugin_id);
            }
            var dest = path.join(targetDir, path.basename(src));

            common.copyNewFile(plugin_dir, src, project_dir, dest);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var dest = path.join(obj.targetDir, path.basename(obj.src));
            common.deleteJava(project_dir, dest);
        }
    },
    'header-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-fileinstall is not supported for amazon-fireos');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.uninstall is not supported for amazon-fireos');
        }
    },
    'lib-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var dest = path.join('libs', path.basename(src));
            common.copyFile(plugin_dir, src, project_dir, dest);
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
            common.copyFile(plugin_dir, src, project_dir, target);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var target = obj.target;
            common.removeFile(project_dir, target);
        }
    },
    'framework': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var custom = obj.custom;
            if (!src) throw new CordovaError('src not specified in <framework> for plugin ' + plugin_id);

            events.emit('verbose', 'Installing Android library: ' + src);
            var parent = obj.parent;
            var parentDir = parent ? path.resolve(project_dir, parent) : project_dir;
            var subDir;

            if (custom) {
                var subRelativeDir = module.exports.getCustomSubprojectRelativeDir(plugin_id, project_dir, src);
                common.copyNewFile(plugin_dir, src, project_dir, subRelativeDir);
                subDir = path.resolve(project_dir, subRelativeDir);
            } else {
                var sdk_dir = module.exports.getProjectSdkDir(project_dir);
                subDir = path.resolve(sdk_dir, src);
            }

            var projectConfig = module.exports.parseProjectFile(project_dir);
            var type = obj.type;
            if (type == 'gradleReference') {
                //add reference to build.gradle
                projectConfig.addGradleReference(parentDir, subDir);
            } else {
                projectConfig.addSubProject(parentDir, subDir);
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

            if (custom) {
                var subRelativeDir = module.exports.getCustomSubprojectRelativeDir(plugin_id, project_dir, src);
                common.removeFile(project_dir, subRelativeDir);
                subDir = path.resolve(project_dir, subRelativeDir);
            } else {
                var sdk_dir = module.exports.getProjectSdkDir(project_dir);
                subDir = path.resolve(sdk_dir, src);
            }

            var projectConfig = module.exports.parseProjectFile(project_dir);
            var type = obj.type;
            if (type == 'gradleReference') {
                projectConfig.removeGradleReference(parentDir, subDir);
            } else {
                projectConfig.removeSubProject(parentDir, subDir);
            }
        }
    },
    parseProjectFile: function(project_dir){
        if (!projectFileCache[project_dir]) {
            projectFileCache[project_dir] = new android_project.AndroidProject();
        }

        return projectFileCache[project_dir];
    },
    purgeProjectFileCache:function(project_dir) {
        delete projectFileCache[project_dir];
    },
    getProjectSdkDir: function (project_dir) {
        var localProperties = properties_parser.createEditor(path.resolve(project_dir, 'local.properties'));
        return localProperties.get('sdk.dir');
    },
    getCustomSubprojectRelativeDir: function (plugin_id, project_dir, src) {
        // All custom subprojects are prefixed with the last portion of the package id.
        // This is to avoid collisions when opening multiple projects in Eclipse that have subprojects with the same name.
        var prefix = module.exports.package_suffix(project_dir);
        var subRelativeDir = path.join(plugin_id, prefix + '-' + path.basename(src));
        return subRelativeDir;
    }
};
