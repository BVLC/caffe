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


var path         = require('path'),
    cordova_util = require('./util'),
    fs           = require('fs'),
    Q            = require('q'),
    child_process = require('child_process');


function getJson(jsonFile) {
    return JSON.parse(fs.readFileSync(jsonFile, 'utf-8'));
}

// Retrieves the platforms and their versions from the platforms.json file
// Returns an array of {platform: platform, version: version} ...
// ... where version could be '3.4.0', '/path/to/platform' or 'git://...'
function getVersions(projectRoot) {
    var platformsDir = path.join(projectRoot, 'platforms');
    var platformsJsonFile = path.join(platformsDir, 'platforms.json');

    // If the platforms.json file doesn't exist, retrieve versions from platforms installed on the filesystem...
    // ...Note that in this case, we won't be able to know what source(folder, git-url) the platform came from, we'll just use versions
    return getPlatVersionsFromFile(platformsJsonFile).fail(function(){
        return getPlatVersionsFromFileSystem(projectRoot);
    });
}

// Returns a promise
function getPlatVersionsFromFile(platformsJsonFile){

    var platformData;

    // Handle 'file not found' exception and stay within the 'promise monad'
    try{
        platformData = getJson(platformsJsonFile);
    } catch(e) {
        return Q.reject(e);
    }

    var platformVersions = [];

    platformVersions = Object.keys(platformData).map(function(p){
        return {platform: p, version: platformData[p]};
    });

    return Q(platformVersions);
}

// Returns a promise
function getPlatVersionsFromFileSystem(projectRoot){
    var platforms_on_fs = cordova_util.listPlatforms(projectRoot);
    var platformVersions = platforms_on_fs.map(function(platform){
        var script = path.join(projectRoot, 'platforms', platform, 'cordova', 'version');
        return Q.ninvoke(child_process, 'exec', script, {}).then(function(result){
            var version = result[0];

            // clean the version we get back from the script
            // This is necessary because the version script uses console.log to pass back
            // the version. Using console.log ends up adding additional line breaks/newlines to the value returned.
            // ToDO: version scripts should be refactored to not use console.log()
            var versionCleaned = version.replace(/\r?\n|\r/g, '');
            return {platform: platform, version: versionCleaned};
        });
    });

    return Q.all(platformVersions);
}

// Saves platform@version into platforms.json
function save(projectRoot, platform, version) {
    var platformsDir = path.join(projectRoot, 'platforms');
    var platformJsonFile = path.join(platformsDir, 'platforms.json');

    var data = {};
    if(fs.existsSync(platformJsonFile)){
        data = getJson(platformJsonFile);
    }
    data[platform] = version;
    fs.writeFileSync(platformJsonFile, JSON.stringify(data, null, 4), 'utf-8');
}

function remove(projectRoot, platform){
    var platformsDir = path.join(projectRoot, 'platforms');
    var platformJsonFile = path.join(platformsDir, 'platforms.json');
    if(!fs.existsSync(platformJsonFile)){
        return;
    }
    var data = getJson(platformJsonFile);
    delete data[platform];
    fs.writeFileSync(platformJsonFile, JSON.stringify(data, null, 4), 'utf-8');
}

module.exports.getPlatformVersions = getVersions;
module.exports.save = save;
module.exports.remove = remove;
