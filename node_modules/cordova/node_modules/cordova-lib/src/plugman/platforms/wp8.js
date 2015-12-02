/*
 *
 * Copyright 2013 Jesse MacFadyen
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

var common = require('./common'),
    path = require('path'),
    glob = require('glob'),
    csproj = require('../../util/windows/csproj'),
    events = require('cordova-common').events,
    xml_helpers = require('cordova-common').xmlHelpers;

module.exports = {
    www_dir:function(project_dir) {
        return path.join(project_dir, 'www');
    },
    package_name:function(project_dir) {
        return xml_helpers.parseElementtreeSync(path.join(project_dir, 'Properties', 'WMAppManifest.xml')).find('App').attrib.ProductID;
    },
    parseProjectFile:function(project_dir) {
        var project_files = glob.sync('*.csproj', {
            cwd:project_dir
        });
        if (project_files.length === 0) {
            throw new Error('does not appear to be a Windows Phone project (no .csproj file)');
        }
        return new csproj(path.join(project_dir, project_files[0]));
    },
    'source-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            var dest = path.join('Plugins', plugin_id, obj.targetDir ? obj.targetDir : '', path.basename(obj.src));

            common.copyNewFile(plugin_dir, obj.src, project_dir, dest);
            // add reference to this file to csproj.
            project_file.addSourceFile(dest);
        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            var dest = path.join('Plugins', plugin_id, obj.targetDir ? obj.targetDir : '', path.basename(obj.src));
            common.removeFile(project_dir, dest);
            // remove reference to this file from csproj.
            project_file.removeSourceFile(dest);
        }
    },
    'header-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.install is not supported for wp8');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.uninstall is not supported for wp8');
        }
    },
    'resource-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'resource-file.install is not supported for wp8');
        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'resource-file.uninstall is not supported for wp8');
        }
    },
    'framework':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'wp8 framework install :: ' + plugin_id  );

            var src = obj.src;
            var dest = src; // if !isCustom, we will just add a reference to the file in place
            var isCustom = obj.custom;

            if(isCustom) {
                dest = path.join('plugins', plugin_id, path.basename(src));
                common.copyFile(plugin_dir, src, project_dir, dest);
            }

            project_file.addReference(dest);

        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'wp8 framework uninstall :: ' + plugin_id  );

            var src = obj.src;
            var isCustom = obj.custom;

            if(isCustom) {
                var dest = path.join('plugins', plugin_id);
                common.removeFile(project_dir, dest);
            }

            project_file.removeReference(src);
        }
    },
    'lib-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'wp8 lib-file install :: ' + plugin_id);
            var inc  = obj.Include;
            project_file.addSDKRef(inc);
        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'wp8 lib-file uninstall :: ' + plugin_id);
            var inc = obj.Include;
            project_file.removeSDKRef(inc);
        }
    }
};
