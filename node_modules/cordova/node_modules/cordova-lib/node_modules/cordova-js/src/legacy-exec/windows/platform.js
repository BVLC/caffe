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
    // for backward compatibility we report 'windows8' when run on Windows 8.0 and 
    // 'windows' for Windows 8.1 and Windows Phone 8.1
    id: (navigator.appVersion.indexOf("MSAppHost/1.0") !== -1) ? 'windows8' : 'windows',
    bootstrap:function() {
        var cordova = require('cordova'),
            exec = require('cordova/exec'),
            channel = cordova.require('cordova/channel'),
            modulemapper = require('cordova/modulemapper');

        modulemapper.clobbers('cordova/exec/proxy', 'cordova.commandProxy');
        channel.onNativeReady.fire();

        var onWinJSReady = function () {
            var app = WinJS.Application;
            var checkpointHandler = function checkpointHandler() {
                cordova.fireDocumentEvent('pause',null,true);
            };

            var resumingHandler = function resumingHandler() {
                cordova.fireDocumentEvent('resume',null,true);
            };

            app.addEventListener("checkpoint", checkpointHandler);
            Windows.UI.WebUI.WebUIApplication.addEventListener("resuming", resumingHandler, false);
            app.start();
        };

        if (!window.WinJS) {
            var scriptElem = document.createElement("script");

            if (navigator.appVersion.indexOf('MSAppHost/3.0') !== -1) {
                // Windows 10 UWP
                scriptElem.src = '/WinJS/js/base.js';
            } else if (navigator.appVersion.indexOf("Windows Phone 8.1;") !== -1) {
                // windows phone 8.1 + Mobile IE 11
                scriptElem.src = "//Microsoft.Phone.WinJS.2.1/js/base.js";
            } else if (navigator.appVersion.indexOf("MSAppHost/2.0;") !== -1) {
                // windows 8.1 + IE 11
                scriptElem.src = "//Microsoft.WinJS.2.0/js/base.js";
            } else {
                // windows 8.0 + IE 10
                scriptElem.src = "//Microsoft.WinJS.1.0/js/base.js";
            }
            scriptElem.addEventListener("load", onWinJSReady);
            document.head.appendChild(scriptElem);
        }
        else {
            onWinJSReady();
        }
    }
};
