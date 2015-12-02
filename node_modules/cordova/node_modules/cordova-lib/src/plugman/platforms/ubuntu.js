/*
 *
 * Copyright 2013 Canonical Ltd.
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

function replaceAt(str, index, char) {
    return str.substr(0, index) + char + str.substr(index + char.length);
}

function toCamelCase(str) {
    return str.split('-').map(function(str) {
        return replaceAt(str, 0, str[0].toUpperCase());
    }).join('');
}

var shell = require('shelljs')
   , fs = require('fs')
   , path = require('path')
   , common = require('./common')
   , events = require('cordova-common').events
   , xml_helpers = require('cordova-common').xmlHelpers;

module.exports = {
    www_dir:function(project_dir) {
        return path.join(project_dir, 'www');
    },

    package_name:function (project_dir) {
        var config_path = path.join(project_dir, 'config.xml');
        var widget_doc = xml_helpers.parseElementtreeSync(config_path);
        return widget_doc._root.attrib['id'];
    },
    'source-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var dest = path.join('build', 'src', 'plugins', plugin_id, path.basename(obj.src));
            common.copyFile(plugin_dir, obj.src, project_dir, dest);

            var cmake = path.join(project_dir, 'build', 'CMakeLists.txt');
            shell.exec('touch ' + cmake);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var dest = path.join(project_dir, 'build', 'src', 'plugins', plugin_id);
            shell.rm(path.join(dest, path.basename(obj.src)));

            var cmake = path.join(project_dir, 'build', 'CMakeLists.txt');
            shell.exec('touch ' + cmake);
        }
    },
    'header-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var dest = path.join('build', 'src', 'plugins', plugin_id, path.basename(obj.src));
            common.copyFile(plugin_dir, obj.src, project_dir, dest);

            var plugins = path.join(project_dir, 'build', 'src', 'coreplugins.cpp');
            var src = String(fs.readFileSync(plugins));

            src = src.replace('INSERT_HEADER_HERE', '#include "plugins/' + plugin_id + '/' + path.basename(obj.src) +'"\nINSERT_HEADER_HERE');
            var class_name = plugin_id.match(/\.[^.]+$/)[0].substr(1);
            class_name = toCamelCase(class_name);
            src = src.replace('INSERT_PLUGIN_HERE', 'INIT_PLUGIN(' + class_name + ');INSERT_PLUGIN_HERE');

            fs.writeFileSync(plugins, src);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var dest = path.join(project_dir, 'build', 'src', 'plugins', plugin_id);
            shell.rm(path.join(dest, path.basename(obj.src)));

            var plugins = path.join(project_dir, 'build', 'src', 'coreplugins.cpp');
            var src = String(fs.readFileSync(plugins));

            src = src.replace('#include "plugins/' + plugin_id + '/' + path.basename(obj.src) +'"', '');
            var class_name = plugin_id.match(/\.[^.]+$/)[0].substr(1);
            class_name = toCamelCase(class_name);
            src = src.replace('INIT_PLUGIN(' + class_name + ');', '');

            fs.writeFileSync(plugins, src);
        }
    },
    'resource-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var dest = path.join('qml', path.basename(obj.src));
            if (obj.targetDir)
                dest = path.join(obj.targetDir, path.basename(obj.src));
            common.copyFile(plugin_dir, obj.src, project_dir, dest);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var dest = path.join(project_dir, 'qml');
            if (obj.targetDir)
                dest = path.join(project_dir, obj.targetDir);
            shell.rm(path.join(dest, path.basename(obj.src)));
        }
    },
    'framework': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'framework.install is not supported for ubuntu');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'framework.uninstall is not supported for ubuntu');
        }
    },
    'lib-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'lib-file.install is not supported for ubuntu');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'lib-file.uninstall is not supported for ubuntu');
        }
    }
};
