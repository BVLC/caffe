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
var childProcess = require('child_process');
var fs           = require('fs');
var path         = require('path');


module.exports = function computeCommitId(callback, cachedGitVersion) {

    if (cachedGitVersion) {
        callback(cachedGitVersion);
        return;
    }
    
    var cordovaJSDir = path.join(__dirname, '../../');
    
    //make sure .git directory exists in cordova.js repo
    if (fs.existsSync(path.join(__dirname, '../../.git'))) {
        var gitPath = 'git';
        var args = 'rev-list HEAD --max-count=1';
        childProcess.exec(gitPath + ' ' + args, {cwd:cordovaJSDir}, function(err, stdout, stderr) {
            var isWindows = process.platform.slice(0, 3) == 'win';
            if (err && isWindows) {
                gitPath = '"' + path.join(process.env['ProgramFiles'], 'Git', 'bin', 'git.exe') + '"';
                childProcess.exec(gitPath + ' ' + args, function(err, stdout, stderr) {
                    if (err) {
                        console.warn('Error during git describe: ' + err);
                        done('???');
                    } else {
                        done(stdout);
                    }
                });
            } else if (err) {
                console.warn('Error during git describe: ' + err);
                done('???');
            } else {
                done(stdout);
            }
        });
    } else {
        //console.log('no git');
        //Can't compute commit ID
        done('???');
    } 

    function done(stdout) {
        var version = stdout.trim();
        cachedGitVersion = version;
        callback(version);
    };
}
