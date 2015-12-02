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

var cordova = require('cordova'),
    base64 = require('cordova/base64');

/**
 * Execute a cordova command.  It is up to the native side whether this action
 * is synchronous or asynchronous.  The native side can return:
 *      Synchronous: PluginResult object as a JSON string
 *      Asynchronous: Empty string ""
 * If async, the native side will cordova.callbackSuccess or cordova.callbackError,
 * depending upon the result of the action.
 *
 * @param {Function} success    The success callback
 * @param {Function} fail       The fail callback
 * @param {String} service      The name of the service to use
 * @param {String} action       Action to be run in cordova
 * @param {String[]} [args]     Zero or more arguments to pass to the method

 */

module.exports = function(success, fail, service, action, args) {

    var callbackId = service + cordova.callbackId++;
    if (typeof success == "function" || typeof fail == "function") {
        cordova.callbacks[callbackId] = {success:success, fail:fail};
    }
    args = args || [];
    // generate a new command string, ex. DebugConsole/log/DebugConsole23/["wtf dude?"]
    for(var n = 0; n < args.length; n++)
    {
        // special case for ArrayBuffer which could not be stringified out of the box
        if(typeof ArrayBuffer !== "undefined" && args[n] instanceof ArrayBuffer)
        {
            args[n] = base64.fromArrayBuffer(args[n]);
        }

        if(typeof args[n] !== "string")
        {
            args[n] = JSON.stringify(args[n]);
        }
    }
    var command = service + "/" + action + "/" + callbackId + "/" + JSON.stringify(args);
    // pass it on to Notify
    try {
        if(window.external) {
            window.external.Notify(command);
        }
        else {
            console.log("window.external not available :: command=" + command);
        }
    }
    catch(e) {
        console.log("Exception calling native with command :: " + command + " :: exception=" + e);
    }
};

