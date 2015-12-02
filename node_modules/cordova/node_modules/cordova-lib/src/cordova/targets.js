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

var cordova_util = require('./util'),
    Q = require('q'),
    superspawn = require('cordova-common').superspawn,
    path = require('path'),
    events = require('cordova-common').events;

function handleError(error) {
    if (error.code === 'ENOENT') {
        events.emit('log', 'Platform does not support ' + this.script);
    } else {
        events.emit('log', 'An unexpected error has occured');
    }
}

function displayDevices(projectRoot, platform, options) {
    var caller = { 'script': 'list-devices' };
    events.emit('log', 'Available ' + platform + ' devices:');
    var cmd = path.join(projectRoot, 'platforms', platform, 'cordova', 'lib', 'list-devices');
    return superspawn.spawn(cmd, options.argv, { stdio: 'inherit', chmod: true }).catch(handleError.bind(caller));
}

function displayVirtualDevices(projectRoot, platform, options) {
    var caller = { 'script': 'list-emulator-images' };
    events.emit('log', 'Available ' + platform + ' virtual devices:');
    var cmd = path.join(projectRoot, 'platforms', platform, 'cordova', 'lib', 'list-emulator-images');
    return superspawn.spawn(cmd, options.argv, { stdio: 'inherit', chmod: true }).catch(handleError.bind(caller));
}

module.exports = function targets(options) {
    var projectRoot = cordova_util.cdProjectRoot();
    options = cordova_util.preProcessOptions(options);

    var result = Q();
    options.platforms.forEach(function(platform) {
        if (options.options.device) {
            result = result.then(displayDevices.bind(null, projectRoot, platform, options.options));
        } else if(options.options.emulator) {
            result = result.then(displayVirtualDevices.bind(null, projectRoot, platform, options.options));
        } else {
            result = result.then(displayDevices.bind(null, projectRoot, platform, options.options))
            .then(displayVirtualDevices.bind(null, projectRoot, platform, options.options));
        }
    });

    return result;
};
