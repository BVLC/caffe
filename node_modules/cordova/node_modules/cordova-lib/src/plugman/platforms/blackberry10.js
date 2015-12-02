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
   ;

var TARGETS = ['device', 'simulator'];

module.exports = {
    www_dir:function(project_dir) {
        return path.join(project_dir, 'www');
    },
    package_name:function(project_dir) {
        var config_path = path.join(module.exports.www_dir(project_dir), 'config.xml');
        var widget_doc = xml_helpers.parseElementtreeSync(config_path);
        return widget_doc._root.attrib['id'];
    },
    'source-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var target = obj.targetDir || plugin_id;
            TARGETS.forEach(function(arch) {
                var dest = path.join('native', arch, 'chrome', 'plugin', target, path.basename(src));

                common.copyNewFile(plugin_dir, src, project_dir, dest);
            });
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var src = obj.src;
            var target = obj.targetDir || plugin_id;
            TARGETS.forEach(function(arch) {
                var dest = path.join('native', arch, 'chrome', 'plugin', target, path.basename(src));
                common.removeFile(project_dir, dest);
            });
        }
    },
    'header-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.install is not supported for blackberry');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.uninstall is not supported for blackberry');
        }
    },
    'lib-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var src = obj.src;
            var arch = obj.arch;
            var dest = path.join('native', arch, 'plugins', 'jnext', path.basename(src));
            common.copyFile(plugin_dir, src, project_dir, dest, false);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var src = obj.src;
            var arch = obj.arch;
            var dest = path.join('native', arch, 'plugins', 'jnext', path.basename(src));
            common.removeFile(project_dir, dest);
        }
    },
    'resource-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'resource-file.install is not supported for blackberry');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'resource-file.uninstall is not supported for blackberry');
        }
    },
    'framework': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'framework.install is not supported for blackberry');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'framework.uninstall is not supported for blackberry');
        }
    }
};
