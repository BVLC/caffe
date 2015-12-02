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

/**
 * Creates a gap bridge iframe used to notify the native code about queued
 * commands.
 */
var cordova = require('cordova'),
    channel = require('cordova/channel'),
    utils = require('cordova/utils'),
    base64 = require('cordova/base64'),
    // XHR mode does not work on iOS 4.2.
    // XHR mode's main advantage is working around a bug in -webkit-scroll, which
    // doesn't exist only on iOS 5.x devices.
    // IFRAME_NAV is the fastest.
    // IFRAME_HASH could be made to enable synchronous bridge calls if we wanted this feature.
    jsToNativeModes = {
        IFRAME_NAV: 0, // Default. Uses a new iframe for each poke.
        // XHR bridge appears to be flaky sometimes: CB-3900, CB-3359, CB-5457, CB-4970, CB-4998, CB-5134
        XHR_NO_PAYLOAD: 1, // About the same speed as IFRAME_NAV. Performance not about the same as IFRAME_NAV, but more variable.
        XHR_WITH_PAYLOAD: 2, // Flakey, and not as performant
        XHR_OPTIONAL_PAYLOAD: 3, // Flakey, and not as performant
        IFRAME_HASH_NO_PAYLOAD: 4, // Not fully baked. A bit faster than IFRAME_NAV, but risks jank since poke happens synchronously.
        IFRAME_HASH_WITH_PAYLOAD: 5, // Slower than no payload. Maybe since it has to be URI encoded / decoded.
        WK_WEBVIEW_BINDING: 6 // Only way that works for WKWebView :)
    },
    bridgeMode,
    execIframe,
    execHashIframe,
    hashToggle = 1,
    execXhr,
    requestCount = 0,
    vcHeaderValue = null,
    commandQueue = [], // Contains pending JS->Native messages.
    isInContextOfEvalJs = 0,
    failSafeTimerId = 0;

function shouldBundleCommandJson() {
    if (bridgeMode === jsToNativeModes.XHR_WITH_PAYLOAD) {
        return true;
    }
    if (bridgeMode === jsToNativeModes.XHR_OPTIONAL_PAYLOAD) {
        var payloadLength = 0;
        for (var i = 0; i < commandQueue.length; ++i) {
            payloadLength += commandQueue[i].length;
        }
        // The value here was determined using the benchmark within CordovaLibApp on an iPad 3.
        return payloadLength < 4500;
    }
    return false;
}

function massageArgsJsToNative(args) {
    if (!args || utils.typeName(args) != 'Array') {
        return args;
    }
    var ret = [];
    args.forEach(function(arg, i) {
        if (utils.typeName(arg) == 'ArrayBuffer') {
            ret.push({
                'CDVType': 'ArrayBuffer',
                'data': base64.fromArrayBuffer(arg)
            });
        } else {
            ret.push(arg);
        }
    });
    return ret;
}

function massageMessageNativeToJs(message) {
    if (message.CDVType == 'ArrayBuffer') {
        var stringToArrayBuffer = function(str) {
            var ret = new Uint8Array(str.length);
            for (var i = 0; i < str.length; i++) {
                ret[i] = str.charCodeAt(i);
            }
            return ret.buffer;
        };
        var base64ToArrayBuffer = function(b64) {
            return stringToArrayBuffer(atob(b64));
        };
        message = base64ToArrayBuffer(message.data);
    }
    return message;
}

function convertMessageToArgsNativeToJs(message) {
    var args = [];
    if (!message || !message.hasOwnProperty('CDVType')) {
        args.push(message);
    } else if (message.CDVType == 'MultiPart') {
        message.messages.forEach(function(e) {
            args.push(massageMessageNativeToJs(e));
        });
    } else {
        args.push(massageMessageNativeToJs(message));
    }
    return args;
}

function iOSExec() {
    if (bridgeMode === undefined) {
        bridgeMode = jsToNativeModes.IFRAME_NAV;
    }

    if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.cordova && window.webkit.messageHandlers.cordova.postMessage) {
        bridgeMode = jsToNativeModes.WK_WEBVIEW_BINDING;
    }

    var successCallback, failCallback, service, action, actionArgs, splitCommand;
    var callbackId = null;
    if (typeof arguments[0] !== "string") {
        // FORMAT ONE
        successCallback = arguments[0];
        failCallback = arguments[1];
        service = arguments[2];
        action = arguments[3];
        actionArgs = arguments[4];

        // Since we need to maintain backwards compatibility, we have to pass
        // an invalid callbackId even if no callback was provided since plugins
        // will be expecting it. The Cordova.exec() implementation allocates
        // an invalid callbackId and passes it even if no callbacks were given.
        callbackId = 'INVALID';
    } else {
        // FORMAT TWO, REMOVED
        try {
            splitCommand = arguments[0].split(".");
            action = splitCommand.pop();
            service = splitCommand.join(".");
            actionArgs = Array.prototype.splice.call(arguments, 1);

            console.log('The old format of this exec call has been removed (deprecated since 2.1). Change to: ' +
                       "cordova.exec(null, null, \"" + service + "\", \"" + action + "\"," + JSON.stringify(actionArgs) + ");"
            );
            return;
        } catch (e) {}
    }

    // If actionArgs is not provided, default to an empty array
    actionArgs = actionArgs || [];

    // Register the callbacks and add the callbackId to the positional
    // arguments if given.
    if (successCallback || failCallback) {
        callbackId = service + cordova.callbackId++;
        cordova.callbacks[callbackId] =
            {success:successCallback, fail:failCallback};
    }

    actionArgs = massageArgsJsToNative(actionArgs);

    var command = [callbackId, service, action, actionArgs];

    if (bridgeMode === jsToNativeModes.WK_WEBVIEW_BINDING) {
        window.webkit.messageHandlers.cordova.postMessage(command);
    } else {
        // Stringify and queue the command. We stringify to command now to
        // effectively clone the command arguments in case they are mutated before
        // the command is executed.
        commandQueue.push(JSON.stringify(command));
    
        // If we're in the context of a stringByEvaluatingJavaScriptFromString call,
        // then the queue will be flushed when it returns; no need for a poke.
        // Also, if there is already a command in the queue, then we've already
        // poked the native side, so there is no reason to do so again.
        if (!isInContextOfEvalJs && commandQueue.length == 1) {
            pokeNative();
        }
    }
}

function pokeNative() {
    switch (bridgeMode) {
    case jsToNativeModes.XHR_NO_PAYLOAD:
    case jsToNativeModes.XHR_WITH_PAYLOAD:
    case jsToNativeModes.XHR_OPTIONAL_PAYLOAD:
        pokeNativeViaXhr();
        break;
    default: // iframe-based.
        pokeNativeViaIframe();
    }
}

function pokeNativeViaXhr() {
    // This prevents sending an XHR when there is already one being sent.
    // This should happen only in rare circumstances (refer to unit tests).
    if (execXhr && execXhr.readyState != 4) {
        execXhr = null;
    }
    // Re-using the XHR improves exec() performance by about 10%.
    execXhr = execXhr || new XMLHttpRequest();
    // Changing this to a GET will make the XHR reach the URIProtocol on 4.2.
    // For some reason it still doesn't work though...
    // Add a timestamp to the query param to prevent caching.
    execXhr.open('HEAD', "/!gap_exec?" + (+new Date()), true);
    if (!vcHeaderValue) {
        vcHeaderValue = /.*\((.*)\)$/.exec(navigator.userAgent)[1];
    }
    execXhr.setRequestHeader('vc', vcHeaderValue);
    execXhr.setRequestHeader('rc', ++requestCount);
    if (shouldBundleCommandJson()) {
        execXhr.setRequestHeader('cmds', iOSExec.nativeFetchMessages());
    }
    execXhr.send(null);
}

function pokeNativeViaIframe() {
    // CB-5488 - Don't attempt to create iframe before document.body is available.
    if (!document.body) {
        setTimeout(pokeNativeViaIframe);
        return;
    }
    if (bridgeMode === jsToNativeModes.IFRAME_HASH_NO_PAYLOAD || bridgeMode === jsToNativeModes.IFRAME_HASH_WITH_PAYLOAD) {
        // TODO: This bridge mode doesn't properly support being removed from the DOM (CB-7735)
        if (!execHashIframe) {
            execHashIframe = document.createElement('iframe');
            execHashIframe.style.display = 'none';
            document.body.appendChild(execHashIframe);
            // Hash changes don't work on about:blank, so switch it to file:///.
            execHashIframe.contentWindow.history.replaceState(null, null, 'file:///#');
        }
        // The delegate method is called only when the hash changes, so toggle it back and forth.
        hashToggle = hashToggle ^ 3;
        var hashValue = '%0' + hashToggle;
        if (bridgeMode === jsToNativeModes.IFRAME_HASH_WITH_PAYLOAD) {
            hashValue += iOSExec.nativeFetchMessages();
        }
        execHashIframe.contentWindow.location.hash = hashValue;
    } else {
        // Check if they've removed it from the DOM, and put it back if so.
        if (execIframe && execIframe.contentWindow) {
            execIframe.contentWindow.location = 'gap://ready';
        } else {
            execIframe = document.createElement('iframe');
            execIframe.style.display = 'none';
            execIframe.src = 'gap://ready';
            document.body.appendChild(execIframe);
        }
        // Use a timer to protect against iframe being unloaded during the poke (CB-7735).
        // This makes the bridge ~ 7% slower, but works around the poke getting lost
        // when the iframe is removed from the DOM.
        // An onunload listener could be used in the case where the iframe has just been
        // created, but since unload events fire only once, it doesn't work in the normal
        // case of iframe reuse (where unload will have already fired due to the attempted
        // navigation of the page).
        failSafeTimerId = setTimeout(function() {
            if (commandQueue.length) {
                pokeNative();
            }
        }, 50); // Making this > 0 improves performance (marginally) in the normal case (where it doesn't fire).
    }
}

iOSExec.jsToNativeModes = jsToNativeModes;

iOSExec.setJsToNativeBridgeMode = function(mode) {
    // Remove the iFrame since it may be no longer required, and its existence
    // can trigger browser bugs.
    // https://issues.apache.org/jira/browse/CB-593
    if (execIframe) {
        if (execIframe.parentNode) {
            execIframe.parentNode.removeChild(execIframe);
        }
        execIframe = null;
    }
    bridgeMode = mode;
};

iOSExec.nativeFetchMessages = function() {
    // Stop listing for window detatch once native side confirms poke.
    if (failSafeTimerId) {
        clearTimeout(failSafeTimerId);
        failSafeTimerId = 0;
    }
    // Each entry in commandQueue is a JSON string already.
    if (!commandQueue.length) {
        return '';
    }
    var json = '[' + commandQueue.join(',') + ']';
    commandQueue.length = 0;
    return json;
};

iOSExec.nativeCallback = function(callbackId, status, message, keepCallback, debug) {
    return iOSExec.nativeEvalAndFetch(function() {
        var success = status === 0 || status === 1;
        var args = convertMessageToArgsNativeToJs(message);
        function nc2() {
            cordova.callbackFromNative(callbackId, success, status, args, keepCallback);
        }
        // CB-8468
        if (debug) {
            setTimeout(nc2, 0);
        } else {
            nc2();
        }
    });
};

iOSExec.nativeEvalAndFetch = function(func) {
    // This shouldn't be nested, but better to be safe.
    isInContextOfEvalJs++;
    try {
        func();
        return iOSExec.nativeFetchMessages();
    } finally {
        isInContextOfEvalJs--;
    }
};

module.exports = iOSExec;
