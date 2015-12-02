/*
 * Licensed to the Apache Software Foundation (ASF
 * or more contributor license agreements.  See th
 * distributed with this work for additional infor
 * regarding copyright ownership.  The ASF license
 * to you under the Apache License, Version 2.0 (t
 * "License"); you may not use this file except in
 * with the License.  You may obtain a copy of the
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to
 * software distributed under the License is distr
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
 * KIND, either express or implied.  See the Licen
 * specific language governing permissions and lim
 * under the License.
 */
var fs           = require('fs');
var path         = require('path');
var browserify   = require('browserify');
var root         = path.join(__dirname, '..', '..');
var pkgJson      = require('../../package.json');
var collectFiles = require('./collect-files');
var copyProps    = require('./copy-props');

module.exports = function bundle(platform, debug, commitId, platformVersion, platformPath) {
    platformPath = fs.existsSync(platformPath) && fs.existsSync(path.join(platformPath, 'cordova-js-src')) ?
        path.join(platformPath, 'cordova-js-src') :
        path.resolve(root, 'src', 'legacy-exec', platform);

    var platformDirname = platform === 'amazon-fireos' ? 'android' : platform;

    var modules = {'cordova': path.resolve(root, 'src', 'cordova_b.js')};
    copyProps(modules, collectFiles(path.resolve(root, 'src', 'common'), 'cordova'));
    copyProps(modules, collectFiles(platformPath, 'cordova'));

    // Replace standart initialization script with browserify's one
    delete modules['cordova/init_b'];
    delete modules['cordova/modulemapper_b'];
    delete modules['cordova/pluginloader_b'];
    modules['cordova/init'] = path.resolve(root, 'src', 'common', 'init_b.js');
    modules['cordova/modulemapper'] = path.resolve(root, 'src', 'common', 'modulemapper_b.js');
    modules['cordova/pluginloader'] = path.resolve(root, 'src', 'common', 'pluginloader_b.js');

    // test doesn't support custom paths
    if (platform === 'test') {
        var testFilesPath;
        var androidPath = path.resolve(pkgJson['cordova-platforms']['cordova-android']);
        var iosPath = path.resolve(pkgJson['cordova-platforms']['cordova-ios']);
        // Add android platform-specific modules that have tests to the test bundle.
        if(fs.existsSync(androidPath)) {
            testFilesPath = path.resolve(androidPath, 'cordova-js-src', 'android');
            modules['cordova/android/exec'] = path.resolve(androidPath, 'cordova-js-src', 'exec.js');
        } else {
            testFilesPath = path.resolve('src', 'legacy-exec', 'android', 'android');
            modules['cordova/android/exec'] = path.resolve(root, 'src', 'legacy-exec', 'android', 'exec.js');
        }
        copyProps(modules, collectFiles(testFilesPath, 'cordova/android'));

        //Add iOS platform-specific modules that have tests for the test bundle.
        if(fs.existsSync(iosPath)) {
            modules['cordova/ios/exec'] = path.join(iosPath, 'cordova-js-src', 'exec.js');
        } else {
            modules['cordova/ios/exec'] = path.resolve(root, 'src', 'legacy-exec', 'ios', 'exec.js');
        }
        copyProps(modules, collectFiles(testFilesPath, 'cordova/ios'));
    }

    modules = Object.keys(modules)
    .map(function (moduleId) {
        return {
            file: modules[moduleId],
            expose: moduleId
        };
    });

    return browserify({debug: !!debug, detectGlobals: false})
        .require(modules)
        .exclude('cordova/plugin_list');
};
