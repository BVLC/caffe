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

/*jshint -W020 */

var utils = require('cordova/utils');
var activeXhrs = [];
var isInstalled = false;
var origXhr = this.XMLHttpRequest;

function MockXhr() {
    this.requestHeaders = {};
    this.readyState = 0;

    this.onreadystatechange = null;
    this.onload = null;
    this.onerror = null;
    this.clearResponse_();
}

MockXhr.prototype.clearResponse_ = function() {
    this.url = null;
    this.method = null;
    this.async = null;
    this.requestPayload = undefined;

    this.statusCode = 0;
    this.responseText = '';
    this.responseHeaders = {};
};

MockXhr.prototype.setReadyState_ = function(value) {
    this.readyState = value;
    this.onreadystatechange && this.onreadystatechange();
};

MockXhr.prototype.open = function(method, url, async) {
    if (this.readyState !== 0 && this.readyState !== 4) {
        throw Error('Tried to open MockXhr while request in progress.');
    }
    this.clearResponse_();
    this.method = method;
    this.url = url;
    this.async = async;
    this.setReadyState_(1);
};

MockXhr.prototype.setRequestHeader = function(key, val) {
    if (this.readyState != 1) {
        throw Error('Tried to setRequestHeader() without call to open()');
    }
    this.requestHeaders[key] = String(val);
};

MockXhr.prototype.send = function(payload) {
    if (this.readyState != 1) {
        throw Error('Tried to send MockXhr without call to open().');
    }
    this.requestPayload = payload;
    this.setReadyState_(2);

    activeXhrs.push(this);
};

MockXhr.prototype.simulateResponse = function(statusCode, responseText, responseHeaders) {
    if (this.readyState != 2) {
        throw Error('Call to simulateResponse() when MockXhr is in state ' + this.readyState);
    }
    for (var i = this.readyState; i <= 4; i++) {
        if (i == 2) {
            this.statusCode = statusCode;
            this.responseHeaders = responseHeaders || this.responseHeaders;
        }
        if (i == 4) {
            this.responseText = responseText;
        }
        this.setReadyState_(i);
    }
    if (statusCode == 200) {
        this.onload && this.onload();
    } else {
        this.onerror && this.onerror();
    }
    utils.arrayRemove(activeXhrs, this);
};

function install() {
    if (isInstalled) {
        throw Error('mockxhr.install called without uninstall.');
    }
    isInstalled = true;
    activeXhrs.length = 0;
    XMLHttpRequest = MockXhr;
}

function uninstall() {
    XMLHttpRequest = origXhr;
    isInstalled = false;
}

module.exports = {
    install: install,
    uninstall: uninstall,
    activeXhrs: activeXhrs
};
