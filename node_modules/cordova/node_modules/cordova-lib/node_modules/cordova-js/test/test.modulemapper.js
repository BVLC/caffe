/*
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

describe('modulemapper', function() {
    var modulemapper = require('cordova/modulemapper'),
        testmodule = require('cordova/test/testmodule'),
        utils = require('cordova/utils');
    var context;

    beforeEach(function() {
        function TestClass() {}
        TestClass.prototype.method1 = function() { return 'orig'; };

        context = {
            func: function() {},
            num: 1,
            obj: { str: 'hello', num: 8, subObj: {num: 9} },
            TestClass: TestClass
        };
        utils.defineGetter(context, 'getme', function() { return 'getter' });
    });

    afterEach(function() {
        modulemapper.reset();
    });

    it('should throw in module does not exist', function() {
        expect(function() { modulemapper.clobbers('cordova/invalid', 'newProp'); }).toThrow();
    });
    it('should properly set a new top-level property', function() {
        modulemapper.clobbers('cordova/test/testmodule', 'newProp1');
        modulemapper.defaults('cordova/test/testmodule', 'newProp2');
        modulemapper.merges('cordova/test/testmodule', 'newProp3');
        modulemapper.mapModules(context);
        expect(context.newProp1).toBe(testmodule);
        expect(context.newProp2).toBe(testmodule);
        expect(context.newProp3).toBe(testmodule);
    });
    it('should properly set a new non-top-level property', function() {
        modulemapper.clobbers('cordova/test/testmodule', 'foo1.newProp');
        modulemapper.defaults('cordova/test/testmodule', 'foo2.newProp');
        modulemapper.merges('cordova/test/testmodule', 'foo3.newProp');
        modulemapper.mapModules(context);
        expect(context.foo1.newProp).toBe(testmodule);
        expect(context.foo2.newProp).toBe(testmodule);
        expect(context.foo3.newProp).toBe(testmodule);
    });
    it('should properly set a new non-top-level property #2', function() {
        modulemapper.clobbers('cordova/test/testmodule', 'foo1.bar.newProp');
        modulemapper.defaults('cordova/test/testmodule', 'foo2.bar.newProp');
        modulemapper.merges('cordova/test/testmodule', 'foo3.bar.newProp');
        modulemapper.mapModules(context);
        expect(context.foo1.bar.newProp).toBe(testmodule);
        expect(context.foo2.bar.newProp).toBe(testmodule);
        expect(context.foo3.bar.newProp).toBe(testmodule);
    });
    it('should properly set a non-new non-top-level property', function() {
        modulemapper.clobbers('cordova/test/testmodule', 'obj.newProp1');
        modulemapper.defaults('cordova/test/testmodule', 'obj.newProp2');
        modulemapper.merges('cordova/test/testmodule', 'obj.newProp3');
        modulemapper.mapModules(context);
        expect(context.obj.newProp1).toBe(testmodule);
        expect(context.obj.newProp2).toBe(testmodule);
        expect(context.obj.newProp3).toBe(testmodule);
    });
    it('should clobber existing properties', function() {
        modulemapper.clobbers('cordova/test/testmodule', 'num');
        modulemapper.clobbers('cordova/test/testmodule', 'obj.str');
        modulemapper.clobbers('cordova/test/testmodule', 'getme');
        modulemapper.clobbers('cordova/test/testmodule', 'TestClass');
        modulemapper.mapModules(context);
        expect(context.num).toBe(testmodule);
        expect(context.obj.str).toBe(testmodule);
        expect(context.getme).toBe(testmodule);
        expect(context.TestClass).toBe(testmodule);
    });
    it('should not clobber existing properties when using defaults', function() {
        modulemapper.defaults('cordova/test/testmodule', 'num');
        modulemapper.defaults('cordova/test/testmodule', 'obj.str');
        modulemapper.defaults('cordova/test/testmodule', 'obj.getme');
        modulemapper.defaults('cordova/test/testmodule', 'TestClass');
        modulemapper.mapModules(context);
        expect(context.num).not.toBe(testmodule);
        expect(context.obj.str).not.toBe(testmodule);
        expect(context.getme).not.toBe(testmodule);
        expect(context.TestClass).not.toBe(testmodule);
    });
    it('should throw when namespace is a non-object', function() {
        expect(function() {
            modulemapper.merges('cordova/test/testmodule', 'num');
            modulemapper.mapModules(context);
        }).toThrow();
    });
    it('should merge into objects', function() {
        modulemapper.merges('cordova/test/testmodule', 'obj');
        modulemapper.mapModules(context);
        for (var k in testmodule) {
            if (k != 'subObj') {
                expect(context.obj[k]).toBe(testmodule[k]);
            }
        }
        expect(context.obj.num).toBe(testmodule.num);
        expect(context.obj.subObj.num).toBe(9);
        expect(context.obj.subObj.str).toBe(testmodule.subObj.str);
    });
    it('should merge into constructor prototypes', function() {
        modulemapper.merges('cordova/test/testmodule', 'TestClass');
        modulemapper.mapModules(context);
        for (var k in testmodule) {
            expect(context.TestClass.prototype[k]).toBe(testmodule[k]);
        }
    });
    it('should maintain order of calls', function() {
        modulemapper.merges('cordova/test/testmodule', 'obj');
        modulemapper.clobbers('cordova/test/testmodule', 'obj');
        modulemapper.mapModules(context);
        expect(context.obj).toBe(testmodule);
    });
    it('should maintain order of calls2', function() {
        modulemapper.merges('cordova/test/testmodule', 'obj.foo');
        modulemapper.clobbers('cordova/test/testmodule', 'obj');
        modulemapper.merges('cordova/test/testmodule', 'obj.obj');
        modulemapper.mapModules(context);
        expect(context.obj.foo).toBeUndefined();
        expect(context.obj).toBe(testmodule);
        expect(context.obj).not.toBe(testmodule.obj);
        expect(context.obj.obj).toBe(testmodule.obj);
    });
    it('should return undefined for getOriginalSymbol("unknown")', function() {
        expect(modulemapper.getOriginalSymbol(context, 'blah')).toBeUndefined();
        modulemapper.mapModules(context);
        expect(modulemapper.getOriginalSymbol(context, 'obj.foo.bar')).toBeUndefined('obj.foo.bar');
    });
    it('should remember original symbols when clobbering', function() {
        var orig = context.obj;
        modulemapper.clobbers('cordova/test/testmodule', 'obj');
        modulemapper.mapModules(context);
        expect(modulemapper.getOriginalSymbol(context, 'obj')).toBe(orig);
    });
    it('should remember original symbols when double clobbering', function() {
        var orig = context.obj;
        modulemapper.clobbers('cordova/test/testmodule', 'obj');
        modulemapper.clobbers('cordova/test/testmodule', 'obj');
        modulemapper.mapModules(context);
        expect(modulemapper.getOriginalSymbol(context, 'obj')).toBe(orig);
    });
    it('should return original symbols when symbol was not clobbered', function() {
        modulemapper.mapModules(context);
        expect(modulemapper.getOriginalSymbol(context, 'obj')).toBe(context.obj);
        expect(modulemapper.getOriginalSymbol(context, 'obj.str')).toBe(context.obj.str);
    });
    it('should log about deprecated property access', function() {
        var origConsoleLog = console.log;
        console.log = jasmine.createSpy('console.log');
        this.after(function() {
            console.log = origConsoleLog;
        });
        modulemapper.clobbers('cordova/test/testmodule', 'obj', 'Use foo instead');
        modulemapper.defaults('cordova/test/testmodule', 'newProp', 'Use foo instead');
        modulemapper.mapModules(context);
        context.obj.func();
        context.obj.func();
        expect(console.log.callCount).toBe(1);
        context.newProp.func();
        context.newProp.func();
        expect(console.log.callCount).toBe(2);
    });
});

