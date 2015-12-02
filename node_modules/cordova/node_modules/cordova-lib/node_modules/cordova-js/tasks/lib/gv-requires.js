#!/usr/bin/env node

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

fs   = require('fs')
path = require('path')

//------------------------------------------------------------------------------
//process.chdir(path.join(__dirname, ".."))

var platforms = getPlatforms()

console.log("//-------------------------------------------------------")
console.log("// graphviz .dot file for Cordova requires by platform")
console.log("// http://www.graphviz.org/")
console.log("// ")
console.log("//   - ./build/gv-requires.js > ~/tmp/requires.dot")
console.log("//   - [edit dot file to leave just one digraph]")
console.log("//   - dot -Tsvg ~/tmp/requires.dot > ~/tmp/requires.svg")
console.log("//   - [open svg file in a browser]")
console.log("//-------------------------------------------------------")
console.log("")

for (var i=0; i<platforms.length; i++) {
    var platform = platforms[i]
    
    generateGraph(platform)
}

//------------------------------------------------------------------------------
function getPlatforms() {
    var entries = fs.readdirSync(path.join(__dirname, '..', '..', "pkg"))
    
    var platforms = []
    
    for (var i=0; i<entries.length; i++) {
        var entry = entries[i]
        
        var match = entry.match(/^cordova\.(.*)\.js$/)
        if (match)
            platforms.push(match[1])
    }
    
    return platforms
}

//------------------------------------------------------------------------------
function generateGraph(platform) {
    var modules = {}
    
    var jsFile = path.join("pkg", "cordova." + platform + ".js")
    
    contents = fs.readFileSync(jsFile, 'utf-8')
    contents = contents.replace(/\n/g, ' ')
    
    modulesSource = contents.split(/define\(/)
    
    console.log("//--------------------------------------------------")
    console.log("// graphviz .dot file for " + platform)
    console.log("//--------------------------------------------------")
    console.log("digraph G {")
    
    for (var i=0; i< modulesSource.length; i++) {
        var moduleSource = modulesSource[i];
        
        var match = moduleSource.match(/'(.*?)'(.*)/)
        if (!match) continue
        
        var moduleName = match[1]
        moduleSource   = match[2]
        
        if (moduleName.match(/\s/)) continue
        if (moduleName   == "")     continue
        if (moduleSource == "")     continue

        modules[moduleName] = modules[moduleName] || []
        // console.log("   found module " + moduleName)
        
        var requires = getRequires(moduleSource, modules[moduleName])
        
        for (var j=0; j < requires.length; j++) {
            var gvModule  =  moduleName.replace(/\//g, '\\n')
            var gvRequire = requires[j].replace(/\//g, '\\n')
            
            console.log('   "' + gvModule + '" -> "' + gvRequire + '";')
        }
        
    }

    console.log("}")
    console.log("")
}

//------------------------------------------------------------------------------
function getRequires(moduleSource, requires) {
    var pattern = /.*?require\((.*?)\)(.*)/

    var result = []
//    console.log(moduleSource)
    
    var match = moduleSource.match(pattern)
    
    while (match) {
        var require  = match[1]
        moduleSource = match[2]
        
        require = require.replace(/'|"/g, '')
        result.push(require)
        
        match = moduleSource.match(pattern)
    }
    
    return result
}

    
