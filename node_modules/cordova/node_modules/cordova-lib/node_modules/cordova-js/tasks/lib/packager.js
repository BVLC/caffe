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
var fs              = require('fs');
var path            = require('path');
var bundle          = require('./bundle');
var computeCommitId = require('./compute-commit-id');


module.exports = function generate(platform, useWindowsLineEndings, platformVersion, platformPath, callback) {
    computeCommitId(function(commitId) {
        var outFile;
        var time = new Date().valueOf();

        var libraryRelease = bundle(platform, false, commitId, platformVersion, platformPath);
        // if we are using windows line endings, we will also add the BOM
        if(useWindowsLineEndings) {
            libraryRelease = "\ufeff" + libraryRelease.split(/\r?\n/).join("\r\n");
        }
        
        time = new Date().valueOf() - time;
        if (!fs.existsSync('pkg')) {
            fs.mkdirSync('pkg');
        }

        outFile = path.join('pkg', 'cordova.' + platform + '.js');
        fs.writeFileSync(outFile, libraryRelease, 'utf8');


        console.log('generated cordova.' + platform + '.js @ ' + commitId + ' in ' + time + 'ms');
        callback();
    });
}
