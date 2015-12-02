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
var fs                 = require('fs');
var path               = require('path');
var util               = require('util');
var bundle             = require('./bundle-browserify');
var computeCommitId    = require('./compute-commit-id');
var writeLicenseHeader = require('./write-license-header');

module.exports = function generate(platform, useWindowsLineEndings, platformVersion, platformPath, done) {
    computeCommitId(function(commitId) {
        var outReleaseFile, outReleaseFileStream,
            outDebugFile, outDebugFileStream,
            releaseBundle, debugBundle;
        var time = new Date().valueOf();

        if (!fs.existsSync('pkg')) {
            fs.mkdirSync('pkg');
        }

        outReleaseFile = path.join('pkg', 'cordova.' + platform + '.js');
        outReleaseFileStream = fs.createWriteStream(outReleaseFile);

        // write license header
        writeLicenseHeader(outReleaseFileStream, platform, commitId, platformVersion);

        bundle(platform, false, commitId, platformVersion, platformPath)
          .add(path.resolve(__dirname, '..', '..', 'src/scripts/bootstrap.js'))
          .bundle()
          .pipe(outReleaseFileStream);

        outReleaseFileStream.on('finish', function() {
          var newtime = new Date().valueOf() - time;
          console.log('generated cordova.' + platform + '.js @ ' + commitId + ' in ' + newtime + 'ms');
          done();
        });
    });
};
