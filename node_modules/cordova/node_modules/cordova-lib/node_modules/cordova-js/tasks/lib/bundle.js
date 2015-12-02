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
var fs           = require('fs');
var path         = require('path');
var collectFiles = require('./collect-files');
var copyProps    = require('./copy-props');
var writeModule  = require('./write-module');
var writeScript  = require('./write-script');
var licensePath  = path.join(__dirname, '..', 'templates', 'LICENSE-for-js-file.txt');
var pkgJson      = require('../../package.json');

module.exports = function bundle(platform, debug, commitId, platformVersion, platformPath) {
    var modules = collectFiles(path.join('src', 'common'));
    var scripts = collectFiles(path.join('src', 'scripts'));
    var platformDep;
    modules[''] = path.join('src', 'cordova.js');

    //check to see if platform has cordova-js-src directory
    if(fs.existsSync(platformPath) && fs.existsSync(path.join(platformPath, 'cordova-js-src'))) {
        copyProps(modules, collectFiles(path.join(platformPath, 'cordova-js-src')));
    } else {
        // for platforms that don't have a release with cordova-js-src yet
        // or if platform === test
        copyProps(modules, collectFiles(path.join('src', 'legacy-exec', platform)));
    }
    //test doesn't support custom paths
    if (platform === 'test') {
        var testFilesPath;
        var androidPath = path.resolve(pkgJson['cordova-platforms']['cordova-android']);
        var iosPath = path.resolve(pkgJson['cordova-platforms']['cordova-ios']);
        // Add android platform-specific modules that have tests to the test bundle.
        if(fs.existsSync(androidPath)) {
            testFilesPath = path.resolve(androidPath, 'cordova-js-src', 'android');
            modules['android/exec'] = path.resolve(androidPath, 'cordova-js-src', 'exec.js');
        } else {
            testFilesPath = path.resolve('src', 'legacy-exec', 'android', 'android');
            modules['android/exec'] = path.resolve('src', 'legacy-exec', 'android', 'exec.js');
        }
        copyProps(modules, collectFiles(testFilesPath, 'android'));

        //Add iOS platform-specific modules that have tests for the test bundle.
        if(fs.existsSync(iosPath)) {
            modules['ios/exec'] = path.join(iosPath, 'cordova-js-src', 'exec.js');
        } else {
            modules['ios/exec'] = path.join('src', 'legacy-exec', 'ios', 'exec.js');
        }
    }

    var output = [];

    output.push("// Platform: " + platform);
    output.push("// "  + commitId);

    // write header
    output.push('/*', fs.readFileSync(licensePath, 'utf8'), '*/');
    output.push(';(function() {');

    output.push("var PLATFORM_VERSION_BUILD_LABEL = '"  + platformVersion + "';");

    // write initial scripts
    if (!scripts['require']) {
        throw new Error("didn't find a script for 'require'")
    }

    writeScript(output, scripts['require'], debug)

    // write modules
    var moduleIds = Object.keys(modules)
    moduleIds.sort()

    for (var i=0; i<moduleIds.length; i++) {
        var moduleId = moduleIds[i]

        writeModule(output, modules[moduleId], moduleId, debug)
    }

    output.push("window.cordova = require('cordova');")

    // write final scripts
    if (!scripts['bootstrap']) {
        throw new Error("didn't find a script for 'bootstrap'")
    }

    writeScript(output, scripts['bootstrap'], debug)

    var bootstrapPlatform = 'bootstrap-' + platform
    if (scripts[bootstrapPlatform]) {
        writeScript(output, scripts[bootstrapPlatform], debug)
    }

    // write trailer
    output.push('})();')

    return output.join('\n')
}

