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

var isLegacy = /(?:web|hpw)OS\/(\d+)/.test(navigator.userAgent);

function LS2Request(uri, params) {
    this.uri = uri;
    params = params || {};
    if(params.method) {
        if(this.uri.charAt(this.uri.length-1) != "/") {
            this.uri += "/";
        }
        this.uri += params.method;
    }
    if(typeof params.onSuccess === 'function') {
        this.onSuccess = params.onSuccess;
    }
    if(typeof params.onFailure === 'function') {
        this.onFailure = params.onFailure;
    }
    if(typeof params.onComplete === 'function') {
        this.onComplete = params.onComplete;
    }
    this.params = (typeof params.parameters === 'object') ? params.parameters : {};
    this.subscribe = params.subscribe || false;
    if(this.subscribe) {
        this.params.subscribe = params.subscribe;
    }
    if(this.params.subscribe) {
        this.subscribe = this.params.subscribe;
    }
    this.resubscribe = params.resubscribe || false;
    this.send();
}

LS2Request.prototype.send = function() {
    if(!window.PalmServiceBridge) {
        console.error("PalmServiceBridge not found.");
        return;
    }
    this.bridge = new PalmServiceBridge();
    var self = this;
    this.bridge.onservicecallback = this.callback = function(msg) {
        var parsedMsg;
        if(self.cancelled) {
            return;
        }
        try {
            parsedMsg = JSON.parse(msg);
        } catch(e) {
            parsedMsg = {
                errorCode: -1,
                errorText: msg
            };
        }
        if((parsedMsg.errorCode || parsedMsg.returnValue===false) && self.onFailure) {
            self.onFailure(parsedMsg);
            if(self.resubscribe && self.subscribe) {
                self.delayID = setTimeout(function() {
                    self.send();
                }, LS2Request.resubscribeDelay);
            }
        } else if(self.onSuccess) {
            self.onSuccess(parsedMsg);
        }
        if(self.onComplete) {
            self.onComplete(parsedMsg);
        }
        if(!self.subscribe) {
            self.cancel();
        }
    };
    this.bridge.call(this.uri, JSON.stringify(this.params));
};

LS2Request.prototype.cancel = function() {
    this.cancelled = true;
    if(this.resubscribeJob) {
        clearTimeout(this.delayID);
    }
    if(this.bridge) {
        this.bridge.cancel();
        this.bridge = undefined;
    }
};

LS2Request.prototype.toString = function() {
    return "[LS2Request]";
};

LS2Request.resubscribeDelay = 10000;

module.exports = {
    request: function (uri, params) {
        var req = new LS2Request(uri, params);
        return req;
    },
    systemPrefix: ((isLegacy) ? "com.palm." : "com.webos."),
    protocol: "luna://"
};
