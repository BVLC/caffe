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

module.exports = {
    id: 'webos',
    bootstrap: function() {
        var channel = require('cordova/channel');
        var isLegacy = /(?:web|hpw)OS\/(\d+)/.test(navigator.userAgent);
        var webOSjsLib = (window.webOS!==undefined);
        if(!webOSjsLib && window.PalmSystem && window.PalmSystem.stageReady && isLegacy) {
            window.PalmSystem.stageReady();
        }
        
        // create global legacy Mojo object if it does not exist
        window.Mojo = window.Mojo || {};

        // Check for support for page visibility api
        if(typeof document.webkitHidden !== "undefined") {
            document.addEventListener("webkitvisibilitychange", function(e) {
                if(document.webkitHidden) {
                    channel.onPause.fire();
                } else {
                    channel.onResume.fire();
                }
            });
        } else { //backward compatability with webOS devices that don't support Page Visibility API
            // LunaSysMgr calls this when the windows is maximized or opened.
            window.Mojo.stageActivated = function() {
                channel.onResume.fire();
            };
            // LunaSysMgr calls this when the windows is minimized or closed.
            window.Mojo.stageDeactivated = function() {
                channel.onPause.fire();
            };
        }

        if(isLegacy && !webOSjsLib) {
            var lp = JSON.parse(PalmSystem.launchParams || "{}") || {};
            window.cordova.fireDocumentEvent("webOSLaunch", {type:"webOSLaunch", detail:lp});
            // LunaSysMgr calls this whenever an app is "launched;"
            window.Mojo.relaunch = function(e) {
                var lp = JSON.parse(PalmSystem.launchParams || "{}") || {};
                if(lp['palm-command'] && lp['palm-command'] == 'open-app-menu') {
                    window.cordova.fireDocumentEvent("menubutton");
                    return true;
                } else {
                    window.cordova.fireDocumentEvent("webOSRelaunch", {type:"webOSRelaunch", detail:lp});
                }
            };
        }
        document.addEventListener("keydown", function(e) {
            // back gesture/button varies by version and build
            if(e.keyCode == 27 || e.keyIdentifier == "U+1200001" ||
                    e.keyIdentifier == "U+001B" || e.keyIdentifier == "Back") {
                window.cordova.fireDocumentEvent("backbutton", e);
            }
        });
        // SmartTV webOS uses HTML5 History API, so bind to that
        // Leave freedom upto developers to enforce History states as they please
        // rather than enforcing particular states
        window.addEventListener("popstate", function(e) {
            window.cordova.fireDocumentEvent("backbutton", e);
        });

        require('cordova/modulemapper').clobbers('cordova/webos/service', 'navigator.service');
        channel.onNativeReady.fire();
    }
};
