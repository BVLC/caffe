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

/*jshint jasmine:true*/

describe('iOS exec', function () {
    var SERVICE = 'TestService';
    var ACTION = 'TestAction';
    var VC_ADDR = '1234';

    var cordova = require('cordova');
    var exec = require('cordova/ios/exec');
    var mockxhr = require('cordova/test/mockxhr');
    var winSpy = jasmine.createSpy('win');
    var failSpy = jasmine.createSpy('fail');
    var origUserAgent = navigator.userAgent;

    beforeEach(function() {
        winSpy.reset();
        failSpy.reset();
        mockxhr.install();
        exec.setJsToNativeBridgeMode(exec.jsToNativeModes.XHR_NO_PAYLOAD);
        navigator.__defineGetter__('userAgent', function(){
            return 'hi there (' + VC_ADDR + ')';
        });
    });

    afterEach(function() {
        expect(mockxhr.activeXhrs.length).toBe(0);
        navigator.__defineGetter__('userAgent', function(){
            return origUserAgent;
        });
    });

    afterEach(mockxhr.uninstall);

    function expectXhr() {
        expect(mockxhr.activeXhrs.length).toBeGreaterThan(0, 'expected an XHR to have been sent.');
        var mockXhr = mockxhr.activeXhrs[0];
        expect(mockXhr.url).toMatch(/^\/!gap_exec\\?/);
        expect(mockXhr.requestHeaders['vc']).toBe(VC_ADDR, 'missing vc header');
        expect(mockXhr.requestHeaders['rc']).toBeDefined('missing rc header.');
        expect(mockXhr.requestHeaders['cmds']).not.toBeDefined();
        expect(mockXhr.requestPayload).toBe(null);
        mockXhr.simulateResponse(200, '');
    }

    function simulateNativeBehaviour(codes) {
        var execPayload = JSON.parse(exec.nativeFetchMessages());
        while (execPayload.length && codes.length) {
            var curPayload = execPayload.shift();
            var callbackId = curPayload[0];
            var moreResults = exec.nativeCallback(callbackId, codes.shift(), 'payload', false);
            if (moreResults) {
                execPayload.push.apply(execPayload, JSON.parse(moreResults));
            }
        }
        expect(codes.length).toBe(0, 'Wrong number of results.');
    }

    describe('exec', function() {
        it('should return "" from nativeFetchMessages work when nothing is pending.', function() {
            var execPayload = exec.nativeFetchMessages();
            expect(execPayload).toBe('');
        });
        it('should work in the win case.', function() {
            exec(winSpy, failSpy, SERVICE, ACTION, []);
            expectXhr();
            simulateNativeBehaviour([1]);
            expect(winSpy).toHaveBeenCalledWith('payload');
            expect(failSpy).not.toHaveBeenCalled();
        });
        it('should work in the fail case.', function() {
            exec(winSpy, failSpy, SERVICE, ACTION, []);
            expectXhr();
            simulateNativeBehaviour([2]);
            expect(winSpy).not.toHaveBeenCalled();
            expect(failSpy).toHaveBeenCalledWith('payload');
        });
        it('should use only one XHR for multiple calls.', function() {
            exec(winSpy, failSpy, SERVICE, ACTION, []);
            exec(winSpy, failSpy, SERVICE, ACTION, []);
            expectXhr();
            simulateNativeBehaviour([1, 2]);
            expect(winSpy).toHaveBeenCalledWith('payload');
            expect(failSpy).toHaveBeenCalledWith('payload');
        });
        it('should send an extra XHR when commands flushed before XHR finishes', function() {
            exec(winSpy, failSpy, SERVICE, ACTION, []);
            simulateNativeBehaviour([1]);
            exec(winSpy, failSpy, SERVICE, ACTION, []);
            expectXhr();
            expectXhr();
            simulateNativeBehaviour([2]);
            expect(winSpy).toHaveBeenCalledWith('payload');
            expect(failSpy).toHaveBeenCalledWith('payload');
        });
        it('should return pending calls made from callbacks.', function() {
            function callMore() {
                exec(winSpy, failSpy, SERVICE, ACTION, []);
                exec(winSpy, failSpy, SERVICE, ACTION, []);
            }
            exec(callMore, failSpy, SERVICE, ACTION, []);
            expectXhr();
            simulateNativeBehaviour([1, 1, 1]);
            expect(winSpy.callCount).toBe(2);
            expect(failSpy).not.toHaveBeenCalled();
        });
    });
});
