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

module.exports = function writeContents(oFile, fileName, contents, debug) {
    
    if (debug) {
        contents += '\n//@ sourceURL=' + fileName
        contents = 'eval(' + JSON.stringify(contents) + ')'
        // this bit makes it easier to identify modules
        // with syntax errors in them
        var handler = 'console.log("exception: in ' + fileName + ': " + e);'
        handler += 'console.log(e.stack);'
        contents = 'try {' + contents + '} catch(e) {' + handler + '}'
    }
    else {
        contents = '// file: ' + fileName.split("\\").join("/") + '\n' + contents;
    }

    oFile.push(contents)
}
