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

/* jshint laxcomma:true, sub:true */

var path = require('path')
    , fs = require('fs')
    , common = require('./common')
    , events = require('cordova-common').events
    , xml_helpers = require('cordova-common').xmlHelpers
    ;

module.exports = {
    www_dir: function(project_dir) {
        return path.join(project_dir, 'www');
    },
    package_name:function(project_dir) {
        // preferred location if cordova >= 3.4
        var preferred_path = path.join(project_dir, 'config.xml');
        var config_path;

        if (!fs.existsSync(preferred_path)) {
            // older location
            var old_config_path = path.join(module.exports.www_dir(project_dir), 'config.xml');
            if (!fs.existsSync(old_config_path)) {
                // output newer location and fail reading
                config_path = preferred_path;
                events.emit('verbose', 'unable to find '+config_path);
            } else {
                config_path = old_config_path;
            }
        } else {
            config_path = preferred_path;
        }
        var widget_doc = xml_helpers.parseElementtreeSync(config_path);
        return widget_doc._root.attrib['id'];
    },
    'source-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            var dest = path.join(obj.targetDir, path.basename(obj.src));
            common.copyFile(plugin_dir, obj.src, project_dir, dest);
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            var dest = path.join(obj.targetDir, path.basename(obj.src));
            common.removeFile(project_dir, dest);
        }
    },
    'header-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-fileinstall is not supported for firefoxos');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.uninstall is not supported for firefoxos');
        }
    },
    'resource-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'resource-file.install is not supported for firefoxos');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'resource-file.uninstall is not supported for firefoxos');
        }
    },
    'framework': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'framework.install is not supported for firefoxos');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'framework.uninstall is not supported for firefoxos');
        }
    },
    'lib-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'lib-file.install is not supported for firefoxos');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'lib-file.uninstall is not supported for firefoxos');
        }
    }
};
