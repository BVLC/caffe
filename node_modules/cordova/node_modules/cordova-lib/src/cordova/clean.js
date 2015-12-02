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
    chain        = require('../util/promise-util').Q_chainmap,
    platform_lib = require('../platforms/platforms');

// Returns a promise.
module.exports = function clean(options) {
    var projectRoot = cordova_util.cdProjectRoot();
    options = cordova_util.preProcessOptions(options);

    var hooksRunner = new HooksRunner(projectRoot);
    return hooksRunner.fire('before_clean', options)
    .then(function () {
        return chain(options.platforms, function (platform) {
            events.emit('verbose', 'Running cleanup for ' + platform + ' platform.');
            return platform_lib
                .getPlatformApi(platform)
                .clean();
        });
    })
    .then(function() {
        return hooksRunner.fire('after_clean', options);
    });
};
