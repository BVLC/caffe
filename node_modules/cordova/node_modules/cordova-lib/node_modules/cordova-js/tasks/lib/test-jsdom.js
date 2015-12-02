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

var path             = require('path');
var fs               = require('fs');
var collect          = require('./collect');
var jas              = require('jasmine-node');
var testLibName      = path.join(__dirname, '..', '..', 'pkg', 'cordova.test.js');
var testLib          = fs.readFileSync(testLibName, 'utf8');

var jsdom    = require("jsdom-no-contextify").jsdom;
var document = jsdom(undefined, { url: 'file:///jsdomtest.info/a?b#c' });
var window   = document.parentWindow;

module.exports = function(callback) {

    console.log('starting node-based tests');

    // put jasmine in scope
    Object.keys(jas).forEach(function (key) {
        this[key] = window[key] = global[key] = jas[key];
    });

    // Hack to fix jsdom with node v0.11.13+
    delete String.prototype.normalize;

    try {
        eval(testLib);
    }
    catch (e) {
        console.log("error eval()ing " + testLibName + ": " + e);
        console.log(e.stack);
        throw e;
    }

    // hijack require
    require = window.cordova.require;
    define  = window.cordova.define;
    // Set up dummy navigator object
    navigator = window.navigator || {};

    // load in our tests
    var tests = [];
    collect(path.join(__dirname, '..', '..', 'test'), tests);
    for (var x in tests) {
        eval(fs.readFileSync(tests[x], "utf-8"));
    }

    var env = jasmine.getEnv();
    env.addReporter(new jas.TerminalReporter({
        color: true,
        onComplete: function(runner) { callback(runner.results().passed()); }
    }));

    console.log("------------");
    console.log("Unit Tests:");
    env.execute();
};
