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
    HooksRunner  = require('../hooks/HooksRunner'),
    events       = require('cordova-common').events,
    Q            = require('q'),
    promiseUtil  = require('../util/promise-util'),
    platform_lib = require('../platforms/platforms');

// Returns a promise.
module.exports = function compile(options) {
    var projectRoot = cordova_util.cdProjectRoot();
    options = cordova_util.preProcessOptions(options);

    var hooksRunner = new HooksRunner(projectRoot);
    return hooksRunner.fire('before_compile', options)
    .then(function () {
        return promiseUtil.Q_chainmap(options.platforms, function (platform) {
            return platform_lib
                .getPlatformApi(platform)
                .build(options.options);
        });
    }).then(function() {
        return hooksRunner.fire('after_compile', options);
    }, function(error) {
        events.emit('log', 'ERROR building one of the platforms: ' + error + '\nYou may not have the required environment or OS to build this project');
        return Q.reject(error);
    });
};
