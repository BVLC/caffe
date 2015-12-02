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

/* jshint node:true, bitwise:true, undef:true, trailing:true, quotmark:true,
 indent:4, unused:vars, latedef:nofunc,
 laxcomma:true, sub:true
 */

var common = require('./common'),
    path = require('path'),
    fs   = require('fs'),
    glob = require('glob'),
    jsprojManager = require('../../util/windows/jsprojManager'),
    events = require('cordova-common').events,
    xml_helpers = require('cordova-common').xmlHelpers;

module.exports = {
    platformName: 'windows',
    InvalidProjectPathError: 'does not appear to be a Windows 8 or Unified Windows Store project (no .projitems|jsproj file)',

    www_dir:function(project_dir) {
        return path.join(project_dir, 'www');
    },
    package_name:function(project_dir) {
        // CB-6976 Windows Universal Apps. To make platform backward compatible
        // with old template we look for package.appxmanifest file as well.
        var manifestPath = fs.existsSync(path.join(project_dir, 'package.windows.appxmanifest')) ?
            path.join(project_dir, 'package.windows.appxmanifest') :
            path.join(project_dir, 'package.appxmanifest');

        var manifest = xml_helpers.parseElementtreeSync(manifestPath);
        return manifest.find('Properties/DisplayName').text;
    },
    parseProjectFile:function(project_dir) {
        var project_files = glob.sync('*.projitems', { cwd:project_dir });
        if (project_files.length === 0) {
            // Windows8.1: for smooth transition and to prevent
            // plugin handling failures we search for old *.jsproj also.
            project_files = glob.sync('*.jsproj', { cwd:project_dir });
            if (project_files.length === 0) {
                throw new Error(this.InvalidProjectPathError);
            }
        }
        return new jsprojManager(path.join(project_dir, project_files[0]));
    },
    'source-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            var targetDir = obj.targetDir || '';
            var dest = path.join('plugins', plugin_id, targetDir, path.basename(obj.src));

            common.copyNewFile(plugin_dir, obj.src, project_dir, dest);
            // add reference to this file to jsproj.
            project_file.addSourceFile(dest);
        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            var dest = path.join('plugins', plugin_id,
                obj.targetDir || '',
                path.basename(obj.src));
            common.removeFile(project_dir, dest);
            // remove reference to this file from csproj.
            project_file.removeSourceFile(dest);
        }
    },
    'header-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.install is not supported for Windows');
        },
        uninstall:function(obj, project_dir, plugin_id, options) {
            events.emit('verbose', 'header-file.uninstall is not supported for Windows');
        }
    },
    'resource-file':{
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            var src = obj.src;
            var dest = obj.target;
            // as per specification resource-file target is specified relative to platform root
            common.copyFile(plugin_dir, src, project_dir, dest);
            project_file.addResourceFileToProject(dest, getTargetConditions(obj));
        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            var dest = obj.target;
            common.removeFile(project_dir, dest);
            project_file.removeResourceFileFromProject(dest, getTargetConditions(obj));
        }
    },
    'lib-file': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            var inc  = obj.Include || obj.src;
            project_file.addSDKRef(inc, getTargetConditions(obj));
        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'windows lib-file uninstall :: ' + plugin_id);
            var inc = obj.Include || obj.src;
            project_file.removeSDKRef(inc, getTargetConditions(obj));
        }
    },
    'framework': {
        install:function(obj, plugin_dir, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'windows framework install :: ' + plugin_id);

            var src = obj.src;
            var dest = src; // if !isCustom, we will just add a reference to the file in place
            // technically it is not possible to get here without isCustom == true -jm
            // var isCustom = obj.custom;
            var type = obj.type;

            if(type === 'projectReference') {
                project_file.addProjectReference(path.join(plugin_dir,src), getTargetConditions(obj));
            }
            else {
                // if(isCustom) {}
                var targetDir = obj.targetDir || '';
                // path.join ignores empty paths passed so we don't check whether targetDir is not empty
                dest = path.join('plugins', plugin_id, targetDir, path.basename(src));
                common.copyFile(plugin_dir, src, project_dir, dest);
                project_file.addReference(dest, getTargetConditions(obj));
            }

        },
        uninstall:function(obj, project_dir, plugin_id, options, project_file) {
            events.emit('verbose', 'windows framework uninstall :: ' + plugin_id  );

            var src = obj.src;
            // technically it is not possible to get here without isCustom == true -jm
            // var isCustom = obj.custom;
            var type = obj.type;

            if(type === 'projectReference') {
                // unfortunately we have to generate the plugin_dir path because it is not passed to uninstall. Note
                // that project_dir is the windows project directory ([project]\platforms\windows) - we need to get to
                // [project]\plugins\[plugin_id]
                var plugin_dir = path.join(project_dir, '..', '..', 'plugins', plugin_id, src);
                project_file.removeProjectReference(plugin_dir, getTargetConditions(obj));
            }
            else {
                // if(isCustom) {  }
                var targetPath = path.join('plugins', plugin_id);
                common.removeFile(project_dir, targetPath);
                project_file.removeReference(src, getTargetConditions(obj));
            }
        }
    }
};

function getTargetConditions(obj) {
    return { versions: obj.versions, deviceTarget: obj.deviceTarget, arch: obj.arch };
}
