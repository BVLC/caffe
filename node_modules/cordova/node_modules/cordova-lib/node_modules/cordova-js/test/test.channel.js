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

describe("channel", function () {
    var channel = require('cordova/channel'),
        multiChannel,
        stickyChannel;

    function callCount(spy) {
        return spy.argsForCall.length;
    }
    function expectCallCount(spy, count) {
        expect(callCount(spy)).toEqual(count);
    }
    beforeEach(function() {
        multiChannel = channel.create('multiChannel');
        stickyChannel = channel.createSticky('stickyChannel');
    });

    describe("subscribe method", function() {
        it("should throw an exception if no function is provided", function() {
            expect(function() {
                multiChannel.subscribe();
            }).toThrow();

            expect(function() {
                multiChannel.subscribe(null);
            }).toThrow();

            expect(function() {
                multiChannel.subscribe(undefined);
            }).toThrow();

            expect(function() {
                multiChannel.subscribe({apply:function(){},call:function(){}});
            }).toThrow();
        });
        it("should not change number of handlers if no function is provided", function() {
            var initialLength = multiChannel.numHandlers;

            try {
                multiChannel.subscribe();
            } catch(e) {}

            expect(multiChannel.numHandlers).toEqual(initialLength);

            try {
                multiChannel.subscribe(null);
            } catch(e) {}

            expect(multiChannel.numHandlers).toEqual(initialLength);
        });
        it("should not change number of handlers when subscribing same function multiple times", function() {
            var handler = function(){};

            multiChannel.subscribe(handler);
            multiChannel.subscribe(handler);
            stickyChannel.subscribe(handler);
            stickyChannel.subscribe(handler);

            expect(multiChannel.numHandlers).toEqual(1);
            expect(stickyChannel.numHandlers).toEqual(1);
        });
    });

    describe("unsubscribe method", function() {
        it("should throw an exception if passed in null or undefined", function() {
            expect(function() {
                multiChannel.unsubscribe();
            }).toThrow();
            expect(function() {
                multiChannel.unsubscribe(null);
            }).toThrow();
        });
        it("should not decrement numHandlers if unsubscribing something that does not exist", function() {
            multiChannel.subscribe(function() {});
            multiChannel.unsubscribe(function() {});
            expect(multiChannel.numHandlers).toEqual(1);
        });
        it("should change the handlers length appropriately", function() {
            var firstHandler = function() {};
            var secondHandler = function() {};
            var thirdHandler = function() {};

            multiChannel.subscribe(firstHandler);
            multiChannel.subscribe(secondHandler);
            multiChannel.subscribe(thirdHandler);
            expect(multiChannel.numHandlers).toEqual(3);

            multiChannel.unsubscribe(thirdHandler);
            expect(multiChannel.numHandlers).toEqual(2);

            multiChannel.unsubscribe(firstHandler);
            multiChannel.unsubscribe(secondHandler);

            expect(multiChannel.numHandlers).toEqual(0);
        });
        it("should not decrement handlers length more than once if unsubscribing a single handler", function() {
            var firstHandler = function(){};
            multiChannel.subscribe(firstHandler);

            expect(multiChannel.numHandlers).toEqual(1);

            multiChannel.unsubscribe(firstHandler);
            multiChannel.unsubscribe(firstHandler);
            multiChannel.unsubscribe(firstHandler);
            multiChannel.unsubscribe(firstHandler);

            expect(multiChannel.numHandlers).toEqual(0);
        });
        it("should not unregister a function registered with a different handler", function() {
            var cHandler = function(){};
            var c2Handler = function(){};
            var c2 = channel.create('jables');
            multiChannel.subscribe(cHandler);
            c2.subscribe(c2Handler);

            expect(multiChannel.numHandlers).toEqual(1);
            expect(c2.numHandlers).toEqual(1);

            multiChannel.unsubscribe(c2Handler);
            c2.unsubscribe(cHandler);

            expect(multiChannel.numHandlers).toEqual(1);
            expect(c2.numHandlers).toEqual(1);
        });
    });

    function commonFireTests(multi) {
        it("should fire all subscribed handlers", function() {
            var testChannel = multi ? multiChannel : stickyChannel;
            var handler = jasmine.createSpy();
            var anotherOne = jasmine.createSpy();

            testChannel.subscribe(handler);
            testChannel.subscribe(anotherOne);

            testChannel.fire();

            expectCallCount(handler, 1);
            expectCallCount(anotherOne, 1);
        });
        it("should pass params to handlers", function() {
            var testChannel = multi ? multiChannel : stickyChannel;
            var handler = jasmine.createSpy();

            testChannel.subscribe(handler);

            testChannel.fire(1, 2, 3);
            expect(handler.argsForCall[0]).toEqual({0:1, 1:2, 2:3});
        });
        it("should not fire a handler that was unsubscribed", function() {
            var testChannel = multi ? multiChannel : stickyChannel;
            var handler = jasmine.createSpy();
            var anotherOne = jasmine.createSpy();

            testChannel.subscribe(handler);
            testChannel.subscribe(anotherOne);
            testChannel.unsubscribe(handler);

            testChannel.fire();

            expectCallCount(handler, 0);
            expectCallCount(anotherOne, 1);
        });
        it("should not fire a handler more than once if it was subscribed more than once", function() {
            var testChannel = multi ? multiChannel : stickyChannel;
            var handler = jasmine.createSpy();

            testChannel.subscribe(handler);
            testChannel.subscribe(handler);
            testChannel.subscribe(handler);

            testChannel.fire();

            expectCallCount(handler, 1);
        });
        it("handler should be called when subscribed, removed, and subscribed again", function() {
            var testChannel = multi ? multiChannel : stickyChannel;
            var handler = jasmine.createSpy();

            testChannel.subscribe(handler);
            testChannel.unsubscribe(handler);
            testChannel.subscribe(handler);

            testChannel.fire();

            expectCallCount(handler, 1);
        });
        it("should not prevent a callback from firing when it is removed during firing.", function() {
            var testChannel = multi ? multiChannel : stickyChannel;
            var handler = jasmine.createSpy().andCallFake(function() { testChannel.unsubscribe(handler2); });
            var handler2 = jasmine.createSpy();
            testChannel.subscribe(handler);
            testChannel.subscribe(handler2);
            testChannel.fire();
            expectCallCount(handler, 1);
            expectCallCount(handler2, 1);
        });
    }
    describe("fire method for sticky channels", function() {
        commonFireTests(false);
        it("should instantly trigger the callback if the event has already been fired", function () {
            var before = jasmine.createSpy('before'),
                after = jasmine.createSpy('after');

            stickyChannel.subscribe(before);
            stickyChannel.fire(1, 2, 3);
            stickyChannel.subscribe(after);

            expectCallCount(before, 1);
            expectCallCount(after, 1);
            expect(after.argsForCall[0]).toEqual({0:1, 1:2, 2:3});
        });
        it("should instantly trigger the callback if the event is currently being fired.", function () {
            var handler1 = jasmine.createSpy().andCallFake(function() { stickyChannel.subscribe(handler2); }),
                handler2 = jasmine.createSpy().andCallFake(function(arg1) { expect(arg1).toEqual('foo');});

            stickyChannel.subscribe(handler1);
            stickyChannel.fire('foo');

            expectCallCount(handler2, 1);
        });
        it("should unregister all handlers after being fired.", function() {
            var handler = jasmine.createSpy();
            stickyChannel.subscribe(handler);
            stickyChannel.fire();
            stickyChannel.fire();
            expectCallCount(handler, 1);
        });
    });
    describe("fire method for multi channels", function() {
        commonFireTests(true);
        it("should not trigger the callback if the event has already been fired", function () {
            var before = jasmine.createSpy('before'),
                after = jasmine.createSpy('after');

            multiChannel.subscribe(before);
            multiChannel.fire();
            multiChannel.subscribe(after);

            expectCallCount(before, 1);
            expectCallCount(after, 0);
        });
        it("should not trigger the callback if the event is currently being fired.", function () {
            var handler1 = jasmine.createSpy().andCallFake(function() { multiChannel.subscribe(handler2); }),
                handler2 = jasmine.createSpy();

            multiChannel.subscribe(handler1);
            multiChannel.fire();
            multiChannel.fire();

            expectCallCount(handler1, 2);
            expectCallCount(handler2, 1);
        });
        it("should not unregister handlers after being fired.", function() {
            var handler = jasmine.createSpy();
            multiChannel.subscribe(handler);
            multiChannel.fire();
            multiChannel.fire();
            expectCallCount(handler, 2);
        });
    });
    describe("channel.join()", function() {
        it("should be called when all functions start unfired", function() {
            var handler = jasmine.createSpy(),
                stickyChannel2 = channel.createSticky('stickyChannel');
            channel.join(handler, [stickyChannel, stickyChannel2]);
            expectCallCount(handler, 0);
            stickyChannel.fire();
            expectCallCount(handler, 0);
            stickyChannel2.fire();
            expectCallCount(handler, 1);
        });
        it("should be called when one functions start fired", function() {
            var handler = jasmine.createSpy(),
                stickyChannel2 = channel.createSticky('stickyChannel');
            stickyChannel.fire();
            channel.join(handler, [stickyChannel, stickyChannel2]);
            expectCallCount(handler, 0);
            stickyChannel2.fire();
            expectCallCount(handler, 1);
        });
        it("should be called when all functions start fired", function() {
            var handler = jasmine.createSpy(),
                stickyChannel2 = channel.createSticky('stickyChannel');
            stickyChannel.fire();
            stickyChannel2.fire();
            channel.join(handler, [stickyChannel, stickyChannel2]);
            expectCallCount(handler, 1);
        });
        it("should throw if a channel is not sticky", function() {
            expect(function() {
                channel.join(function(){}, [stickyChannel, multiChannel]);
            }).toThrow();
        });
    });
    describe("onHasSubscribersChange", function() {
        it("should be called only when the first subscriber is added and last subscriber is removed.", function() {
            var handler = jasmine.createSpy().andCallFake(function() {
                if (callCount(handler) == 1) {
                    expect(this.numHandlers).toEqual(1);
                } else {
                    expect(this.numHandlers).toEqual(0);
                }
            });
            multiChannel.onHasSubscribersChange = handler;
            function foo1() {}
            function foo2() {}
            multiChannel.subscribe(foo1);
            multiChannel.subscribe(foo2);
            multiChannel.unsubscribe(foo1);
            multiChannel.unsubscribe(foo2);
            expectCallCount(handler, 2);
        });
    });
});
