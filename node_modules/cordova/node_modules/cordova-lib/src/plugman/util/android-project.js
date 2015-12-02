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
/*
    Helper for Android projects configuration
*/

/* jshint boss:true */

var fs = require('fs'),
    path = require('path'),
    properties_parser = require('properties-parser'),
    shell = require('shelljs');


function addToPropertyList(projectProperties, key, value) {
    var i = 1;
    while (projectProperties.get(key + '.' + i))
        i++;

    projectProperties.set(key + '.' + i, value);
    projectProperties.dirty = true;
}

function removeFromPropertyList(projectProperties, key, value) {
    var i = 1;
    var currentValue;
    while (currentValue = projectProperties.get(key + '.' + i)) {
        if (currentValue === value) {
            while (currentValue = projectProperties.get(key + '.' + (i + 1))) {
                projectProperties.set(key + '.' + i, currentValue);
                i++;
            }
            projectProperties.set(key + '.' + i);
            break;
        }
        i++;
    }
    projectProperties.dirty = true;
}

function AndroidProject(projectDir) {
    this._propertiesEditors = {};
    this._subProjectDirs = {};
    this._dirty = false;
    this.projectDir = projectDir;
    this.needsSubLibraryUpdate = false;
}

AndroidProject.prototype = {
    addSubProject: function(parentDir, subDir) {
        var parentProjectFile = path.resolve(parentDir, 'project.properties');
        var subProjectFile = path.resolve(subDir, 'project.properties');
        var parentProperties = this._getPropertiesFile(parentProjectFile);
        // TODO: Setting the target needs to happen only for pre-3.7.0 projects
        if (fs.existsSync(subProjectFile)) {
            var subProperties = this._getPropertiesFile(subProjectFile);
            subProperties.set('target', parentProperties.get('target'));
            subProperties.dirty = true;
            this._subProjectDirs[subDir] = true;
        }
        addToPropertyList(parentProperties, 'android.library.reference', module.exports.getRelativeLibraryPath(parentDir, subDir));

        this._dirty = true;
    },
    removeSubProject: function(parentDir, subDir) {
        var parentProjectFile = path.resolve(parentDir, 'project.properties');
        var parentProperties = this._getPropertiesFile(parentProjectFile);
        removeFromPropertyList(parentProperties, 'android.library.reference', module.exports.getRelativeLibraryPath(parentDir, subDir));
        delete this._subProjectDirs[subDir];
        this._dirty = true;
    },
    addGradleReference: function(parentDir, subDir) {
        var parentProjectFile = path.resolve(parentDir, 'project.properties');
        var parentProperties = this._getPropertiesFile(parentProjectFile);
        addToPropertyList(parentProperties, 'cordova.gradle.include', module.exports.getRelativeLibraryPath(parentDir, subDir));
        this._dirty = true;
    },
    removeGradleReference: function(parentDir, subDir) {
        var parentProjectFile = path.resolve(parentDir, 'project.properties');
        var parentProperties = this._getPropertiesFile(parentProjectFile);
        removeFromPropertyList(parentProperties, 'cordova.gradle.include', module.exports.getRelativeLibraryPath(parentDir, subDir));
        this._dirty = true;
    },
    addSystemLibrary: function(parentDir, value) {
        var parentProjectFile = path.resolve(parentDir, 'project.properties');
        var parentProperties = this._getPropertiesFile(parentProjectFile);
        addToPropertyList(parentProperties, 'cordova.system.library', value);
        this._dirty = true;
    },
    removeSystemLibrary: function(parentDir, value) {
        var parentProjectFile = path.resolve(parentDir, 'project.properties');
        var parentProperties = this._getPropertiesFile(parentProjectFile);
        removeFromPropertyList(parentProperties, 'cordova.system.library', value);
        this._dirty = true;
    },
    write: function() {
        if (!this._dirty) {
            return;
        }
        this._dirty = false;

        for (var filename in this._propertiesEditors) {
            var editor = this._propertiesEditors[filename];
            if (editor.dirty) {
                fs.writeFileSync(filename, editor.toString());
                editor.dirty = false;
            }
        }

        // Starting with 3.6.0, the build scripts set ANDROID_HOME, so there is
        // no reason to keep run this command. Plus - we really want to avoid
        // relying on the presense of native SDKs within plugman.
        if (this.needsSubLibraryUpdate) {
            for (var sub_dir in this._subProjectDirs)
            {
                shell.exec('android update lib-project --path "' + sub_dir + '"');
            }
        }
        this._dirty = false;
    },
    _getPropertiesFile: function (filename) {
        if (!this._propertiesEditors[filename]) {
            if (fs.existsSync(filename)) {
                this._propertiesEditors[filename] = properties_parser.createEditor(filename);
            } else {
                this._propertiesEditors[filename] = properties_parser.createEditor();
            }
        }

        return this._propertiesEditors[filename];
    }
};


module.exports = {
    AndroidProject: AndroidProject,
    getRelativeLibraryPath: function (parentDir, subDir) {
        var libraryPath = path.relative(parentDir, subDir);
        return (path.sep == '\\') ? libraryPath.replace(/\\/g, '/') : libraryPath;
    }
};
