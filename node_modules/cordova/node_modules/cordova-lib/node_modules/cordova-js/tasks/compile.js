/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
*/
var generate = require('./lib/packager');
var fs = require('fs');
var path = require('path');
var pkgJson = require('../package.json');

module.exports = function(grunt) {
    grunt.registerMultiTask('compile', 'Packages cordova.js', function() {
        var done = this.async();
        var platformName = this.target;
        var useWindowsLineEndings = this.data.useWindowsLineEndings;
       
        //grabs --platformVersion flag
        var flags = grunt.option.flags();
        var platformVersion;
        var platformPath = undefined;
        flags.forEach(function(flag) {
            //see if --platformVersion was passed in
            if (flag.indexOf('platformVersion') > -1) {
                var equalIndex = flag.indexOf('=');
                platformVersion = flag.slice(equalIndex + 1);
            }
            
            //see if flags for platforms were passed in
            //followed by custom paths
            if (flag.indexOf(platformName) > -1) {
                var equalIndex = flag.indexOf('=');
                platformPath = flag.slice(equalIndex + 1);
            }
        });
        //Use platformPath from package.json, no custom platform path
        if(platformPath === undefined) { 
            platformPath = pkgJson['cordova-platforms']['cordova-'+platformName];
        }
        //Get absolute path to platform
        if(platformPath) {
            platformPath = path.resolve(platformPath);
        }
        if(!platformVersion) {
            var platformPkgJson;

            if(platformPath && fs.existsSync(platformPath)) {
                platformPkgJson = require(platformPath +'/package.json');
                platformVersion = platformPkgJson.version;
            } else {
                platformVersion="N/A";
            }
        }
        generate(platformName, useWindowsLineEndings, platformVersion, platformPath, done);
    });
}
