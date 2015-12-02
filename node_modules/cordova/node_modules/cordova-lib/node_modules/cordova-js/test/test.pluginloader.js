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

describe('pluginloader', function() {
    var pluginloader = require('cordova/pluginloader');
    var injectScript;
    var cdvScript;
    var done;
    var success;
    beforeEach(function() {
        injectScript = spyOn(pluginloader, 'injectScript');
        var el = document.createElement('script');
        el.setAttribute('type', 'foo');
        el.src = 'foo/cordova.js?bar';
        document.body.appendChild(el);
        cdvScript = el;
        done = false;
        success = false;
    });
    afterEach(function() {
        if (cdvScript) {
            cdvScript.parentNode.removeChild(cdvScript);
            cdvScript = null;
        }
        define.remove('cordova/plugin_list');
        define.remove('some.id');
    });

    function setDone() {
        done = true;
    }

    it('should inject cordova_plugins.js when it is not already there', function() {
        injectScript.andCallFake(function(url, onload, onerror) {
            // jsdom deficiencies:
            if (typeof location != 'undefined') {
                expect(url).toBe(window.location.href.replace(/\/[^\/]*?$/, '/foo/cordova_plugins.js'));
            } else {
                expect(url).toBe('foo/cordova_plugins.js');
            }
            define('cordova/plugin_list', function(require, exports, module) {
                module.exports = [];
            });
            success = true;
            onload();
        });

        pluginloader.load(setDone);
        waitsFor(function() { return done });
        runs(function() {
            expect(success).toBe(true);
        });
    });

    it('should not inject cordova_plugins.js when it is already there', function() {
        define('cordova/plugin_list', function(require, exports, module) {
            module.exports = [];
        });
        pluginloader.load(setDone);
        waitsFor(function() { return done });
        runs(function() {
            expect(injectScript).not.toHaveBeenCalled();
        });
    });

    it('should inject plugin scripts when they are not already there', function() {
        define('cordova/plugin_list', function(require, exports, module) {
            module.exports = [
                { "file": "some/path.js", "id": "some.id" }
            ];
        });
        injectScript.andCallFake(function(url, onload, onerror) {
            // jsdom deficiencies:
            if (typeof location != 'undefined') {
                expect(url).toBe(window.location.href.replace(/\/[^\/]*?$/, '/foo/some/path.js'));
            } else {
                expect(url).toBe('foo/some/path.js');
            }
            define('some.id', function(require, exports, module) {
            });
            success = true;
            onload();
        });
        pluginloader.load(setDone);
        waitsFor(function() { return done });
        runs(function() {
            expect(success).toBe(true);
        });
    });

    it('should not inject plugin scripts when they are already there', function() {
        define('cordova/plugin_list', function(require, exports, module) {
            module.exports = [
                { "file": "some/path.js", "id": "some.id" }
            ];
        });
        define('some.id', function(require, exports, module) {
        });
        pluginloader.load(setDone);
        waitsFor(function() { return done });
        runs(function() {
            expect(injectScript).not.toHaveBeenCalled();
        });
    });
});
