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
var fs = require('fs');
var path = require('path');
var pkgJson = require(process.cwd() + '/package.json');

try {
    require('browserify');
} catch (e) {
    console.error("\nbrowserify is not installed, you need to:\n\trun `npm install` from " + require('path').dirname(__dirname)+"\n");
    process.exit(1);
}

var generate = require('./lib/packager-browserify');

module.exports = function(grunt) {
    grunt.registerMultiTask('compile-browserify', 'Packages cordova.js browserify style', function() {
        var done = this.async();
        var platformName = this.target;
        var useWindowsLineEndings = this.data.useWindowsLineEndings;

        //grabs --platformVersion flag
        var flags = grunt.option.flags();
        var platformVersion;
        flags.forEach(function(flag) {
            if (flag.indexOf('platformVersion') > -1) {
                var equalIndex = flag.indexOf('=');
                platformVersion = flag.slice(equalIndex + 1);
            }
        });
        if(!platformVersion){
            var platformPath = pkgJson['cordova-platforms']['cordova-'+platformName];
            if(platformPath && fs.existsSync(platformPath)) {
                var platformPkgJson = require('../' + platformPath +'/package.json');
                platformVersion = platformPkgJson.version;
            } else {
                console.log('platformVersion not supplied. Setting platformVersion to N/A');
                console.log('ex: grunt compile-browserify --platformVersion=3.6.0');
                platformVersion = 'N/A';
            }
        }
        generate(platformName, useWindowsLineEndings, platformVersion, platformPath, done);
    });
}
