/*
 *
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
 *
*/


var fs       = require('fs');
var path     = require('path');
var connect  = require('connect');
var bundle   = require('./bundle');
var collect  = require('./collect');
var start    = require('open');

var testLibName    = path.join(__dirname, '..', '..', 'pkg', 'cordova.test.js');
var testLib        = fs.readFileSync(testLibName, 'utf8');

var pathToTemplate = path.join(__dirname, '..', 'templates', 'suite.html');
var pathToVendor   = path.join(__dirname, '..', 'vendor');
var pathToJasmine  = path.join(__dirname, '..', '..', 'node_modules', 'jasmine-node', 'lib', 'jasmine-node');
var pathToTests    = path.join(__dirname, '..', '..', 'test');

var template = fs.readFileSync(pathToTemplate, "utf-8");

// middlewar for GET '/cordova.test.js'
function cordovajs(req, res) {
    res.writeHead(200, {
        "Cache-Control": "no-cache",
        "Content-Type": "text/javascript"
    });
    res.end(testLib);
}

// middleware for GET '/'
function root(req, res) {
    res.writeHead(200, {
        "Cache-Control": "no-cache",
        "Content-Type": "text/html"
    });

    //FIXME in place collect thing is atrocious
    //create the script tags to include
    var tests = [];
    collect(path.join(__dirname, '..', '..', 'test'), tests);
    var specs = tests.map(function (file, path) {
        return '<script src="' + file.replace(/\\/g, '/').replace(/^.*\/test\//, "/") +
            '" type="text/javascript" charset="utf-8"></script>';
    }).join('\n');

    template = template.replace(/<!-- ##TESTS## -->/g, specs);

    // write the document
    res.end(template);
}

// connect router defn
function routes(app) {
    app.get('/cordova.test.js', cordovajs);
    app.get('/', root);
}

module.exports = function() {
    console.log('starting browser-based tests');

    var vendor = connect.static(pathToVendor);
    var jasmine = connect.static(pathToJasmine);
    var tests  = connect.static(pathToTests);
    var router = connect.router(routes);

    connect(vendor, jasmine, tests, router).listen(3000);

    console.log("Test Server running on:\n");
    console.log("http://127.0.0.1:3000\n");

    start('http://127.0.0.1:3000');
};

