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
    execProxy = require('cordova/exec/proxy');

function RemoteFunctionCall(functionUri) {
    var params = {};

    function composeUri() {
        return "http://localhost:8472/" + functionUri;
    }

    function createXhrRequest(uri, isAsync) {
        var request = new XMLHttpRequest();
        request.open("POST", uri, isAsync);
        request.setRequestHeader("Content-Type", "application/json");
        return request;
    }

    this.addParam = function (name, value) {
        params[name] = encodeURIComponent(JSON.stringify(value));
    };

    this.makeAsyncCall = function () {
        var requestUri = composeUri(),
            request = new XMLHttpRequest(),
            didSucceed,
            response,
            fail = function () {
                var callbackId = JSON.parse(decodeURIComponent(params.callbackId));
                response = JSON.parse(decodeURIComponent(request.responseText) || "null");
                cordova.callbacks[callbackId].fail && cordova.callbacks[callbackId].fail(response.msg, response);
                delete cordova.callbacks[callbackId];
            };

        request.open("POST", requestUri, true /* async */);
        request.setRequestHeader("Content-Type", "application/json");
        request.timeout = 1000; // Timeout in 1000ms
        request.ontimeout = fail;
        request.onerror = fail;

        request.onload = function () {
            response = JSON.parse(decodeURIComponent(request.responseText) || "null");
            if (request.status === 200) {
                didSucceed = response.code === cordova.callbackStatus.OK || response.code === cordova.callbackStatus.NO_RESULT;
                cordova.callbackFromNative(
                        JSON.parse(decodeURIComponent(params.callbackId)),
                        didSucceed,
                        response.code,
                        [ didSucceed ? response.data : response.msg ],
                        !!response.keepCallback
                        );
            } else {
                fail();
            }
        };

        request.send(JSON.stringify(params));
    };

    this.makeSyncCall = function () {
        var requestUri = composeUri(),
        request = createXhrRequest(requestUri, false),
        response;
        try {
            request.send(JSON.stringify(params));
            response = JSON.parse(decodeURIComponent(request.responseText) || "null");
        } catch (e) {
            console.error('makeSyncCall failed', e);
            response = { code: 500, msg: 'exec bridge failure' };
        }
        return response;
    };

}

module.exports = function (success, fail, service, action, args, sync) {
    var uri = service + "/" + action,
    request = new RemoteFunctionCall(uri),
    callbackId = service + cordova.callbackId++,
    proxy,
    response,
    name,
    didSucceed;

    cordova.callbacks[callbackId] = {
        success: success,
        fail: fail
    };

    proxy = execProxy.get(service, action);

    if (proxy) {
        proxy(success, fail, args);
    }

    else {

        request.addParam("callbackId", callbackId);

        for (name in args) {
            if (Object.hasOwnProperty.call(args, name)) {
                request.addParam(name, args[name]);
            }
        }

        if (sync !== undefined && !sync) {
            request.makeAsyncCall();
            return;
        }

        response = request.makeSyncCall();

        if (response.code < 0) {
            if (fail) {
                fail(response.msg, response);
            }
            delete cordova.callbacks[callbackId];
        } else {
            didSucceed = response.code === cordova.callbackStatus.OK || response.code === cordova.callbackStatus.NO_RESULT;
            cordova.callbackFromNative(
                callbackId,
                didSucceed,
                response.code,
                [ didSucceed ? response.data : response.msg ],
                !!response.keepCallback
            );
        }
    }

};
