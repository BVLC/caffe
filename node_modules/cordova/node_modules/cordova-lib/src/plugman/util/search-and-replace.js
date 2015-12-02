#!/usr/bin/env node
/*
 *
 * Copyright 2013 Brett Rudd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
*/

var glob = require('glob'),
    fs = require('fs');

module.exports = searchAndReplace;
function searchAndReplace(srcGlob, variables) {
    var files = glob.sync(srcGlob);
    for (var i in files) {
        var file = files[i];
        if (fs.lstatSync(file).isFile()) {
            var contents = fs.readFileSync(file, 'utf-8');
            for (var key in variables) {
                var regExp = new RegExp('\\$' + key, 'g');
                contents = contents.replace(regExp, variables[key]);
            }
            fs.writeFileSync(file, contents);
        }
    }
}
