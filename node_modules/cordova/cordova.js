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

// All cordova js API moved to cordova-lib. If you don't need the cordova CLI,
// use cordova-lib directly.

var cordova_lib = require('cordova-lib');
module.exports = cordova_lib.cordova;

// Also export the cordova-lib so that downstream consumers of cordova lib and
// CLI will be able to use CLI's cordova-lib and avoid the risk of having two
// different versions of cordova-lib which would result in two instances of
// "events" and can cause bad event handling.
module.exports.cordova_lib = cordova_lib;
module.exports.cli = require('./src/cli');
