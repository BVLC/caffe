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

    id: "blackberry10",

    bootstrap: function() {

        var channel = require('cordova/channel'),
            addEventListener = document.addEventListener;

        //ready as soon as the plugins are
        channel.onPluginsReady.subscribe(function () {
            channel.onNativeReady.fire();
        });

        //pass document online/offline event listeners to window
        document.addEventListener = function (type) {
            if (type === "online" || type === "offline") {
                window.addEventListener.apply(window, arguments);
            } else {
                addEventListener.apply(document, arguments);
            }
        };

        //map blackberry.event to document
        if (!window.blackberry) {
            window.blackberry = {};
        }
        window.blackberry.event =
        {
            addEventListener: document.addEventListener,
            removeEventListener: document.removeEventListener
        };

    }

};
