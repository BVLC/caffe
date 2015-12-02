/**
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
'License'); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

/*
A utility funciton to help output the information needed
when submitting a help request.
Outputs to a file
 */
var cordova_util = require('./util'),
    superspawn   = require('cordova-common').superspawn,
    package      = require('../../package'),
    path         = require('path'),
    fs           = require('fs'),
    Q            = require('q');

// Execute using a child_process exec, for any async command
function execSpawn(command, args, resultMsg, errorMsg) {
    return superspawn.spawn(command, args).then(function(result) {
        return resultMsg + result;
    }, function(error) {
        return errorMsg + error;
    });
}

function getPlatformInfo(platform, projectRoot) {
    switch (platform) {
    case 'ios':
        return execSpawn('xcodebuild', ['-version'], 'iOS platform:\n\n', 'Error retrieving iOS platform information: ');
    case 'android':
        return execSpawn('android', ['list', 'target'], 'Android platform:\n\n', 'Error retrieving Android platform information: ');
    }
}


module.exports = function info() {
    //Get projectRoot
    var projectRoot = cordova_util.cdProjectRoot();
    var output = '';
    if (!projectRoot) {
        return Q.reject(new Error('Current working directory is not a Cordova-based project.'));
    }

    //Array of functions, Q.allSettled
    console.log('Collecting Data...\n\n');
    return Q.allSettled([
            //Get Node version
            Q('Node version: ' + process.version),
            //Get Cordova version
            Q('Cordova version: ' + package.version),
            //Get project config.xml file using ano
            getProjectConfig(projectRoot),
            //Get list of plugins
            listPlugins(projectRoot),
            //Get Platforms information
            getPlatforms(projectRoot)
        ]).then(function(promises) {
            promises.forEach(function(p) {
                output += p.state === 'fulfilled' ? p.value + '\n\n' : p.reason + '\n\n';
            });
            console.info(output);
            fs.writeFile(path.join(projectRoot, 'info.txt'), output, 'utf-8', function (err) {
                if (err) throw err;
            });
        });
};

function getPlatforms(projectRoot) {
    var platforms = cordova_util.listPlatforms(projectRoot);
    if (platforms.length) {
        return Q.all(platforms.map(function(p) {
            return getPlatformInfo(p, projectRoot);
        })).then(function(outs) {
            return outs.join('\n\n');
        });
    }
    return Q.reject('No Platforms Currently Installed');
}

function listPlugins(projectRoot) {
    var pluginPath = path.join(projectRoot, 'plugins'),
        plugins    = cordova_util.findPlugins(pluginPath);

    if (!plugins.length) {
        return Q.reject('No Plugins Currently Installed');
    }
    return Q('Plugins: \n\n' + plugins);
}

function getProjectConfig(projectRoot) {
    if (!fs.existsSync(projectRoot)  ) {
        return Q.reject('Config.xml file not found');
    }
    return Q('Config.xml file: \n\n' + (fs.readFileSync(cordova_util.projectConfig(projectRoot), 'utf-8')));
}
