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
var fs          = require('fs');
var path        = require('path');
var copyProps   = require('./copy-props');
var getModuleId = require('./get-module-id');


function collectFiles(dir, id) {
    if (!id) id = ''

    var result = {}    
    var entries = fs.readdirSync(dir)

    entries = entries.filter(function(entry) {
        if (entry.match(/\.js$/)) 
            return true
        
        var stat = fs.statSync(path.join(dir, entry))

        if (stat.isDirectory())  
            return true
    })

    entries.forEach(function(entry) {
        var moduleId = (id ? id + '/' : '') + entry;
        var fileName = path.join(dir, entry)
        
        var stat = fs.statSync(fileName)
        if (stat.isDirectory()) {
            copyProps(result, collectFiles(fileName, moduleId))
        }
        else {
            moduleId         = getModuleId(moduleId)
            result[moduleId] = fileName
        }
    })
    return copyProps({}, result)
}

module.exports = collectFiles;
