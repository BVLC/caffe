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
var fs            = require('fs');
var path          = require('path');
var stripHeader   = require('./strip-header');
var writeContents = require('./write-contents');


module.exports = function writeModule(oFile, fileName, moduleId, debug) {
    var contents = fs.readFileSync(fileName, 'utf8')

    contents = '\n' + stripHeader(contents, fileName) + '\n'

	// Windows fix, '\' is an escape, but defining requires '/' -jm
    moduleId = path.join('cordova', moduleId).split("\\").join("/");
    
    var signature = 'function(require, exports, module)';
    
    contents = 'define("' + moduleId + '", ' + signature + ' {' + contents + '});\n'

    writeContents(oFile, fileName, contents, debug)    
}

