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

describe('argscheck', function() {
    var argscheck = require('cordova/argscheck');

    function createTestFunc(allowNull) {
        return function testFunc(num, obj, arr, str, date, func) {
            var spec = allowNull ? 'NOASDF*' : 'noasdf*';
            argscheck.checkArgs(spec, 'testFunc', arguments);
        };
    }
    afterEach(function() {
      argscheck.enableChecks = true;
    });

    it('should not throw when given valid args', function() {
        var testFunc = createTestFunc(false);
        testFunc(0, {}, [], '', new Date(), testFunc, 1);
    });
    it('should not throw when given valid optional args', function() {
        var testFunc = createTestFunc(true);
        testFunc(0, {}, [], '', new Date(), testFunc, '');
    });
    it('should not throw when given missing optional args', function() {
        var testFunc = createTestFunc(true);
        testFunc();
    });
    it('should not throw when given null optional args', function() {
        var testFunc = createTestFunc(true);
        testFunc(null, null, null, null, null, null, null);
    });
    it('should throw when given invalid number', function() {
        var testFunc = createTestFunc(true);
        expect(function() { testFunc('foo', null, null, null, null, null) }).toThrow('Wrong type for parameter "num" of testFunc: Expected Number, but got String.');
    });
    it('should throw when given invalid object', function() {
        var testFunc = createTestFunc(true);
        // Do not allow arrays for objects since we're usually dealing with JSON when expecting objects.
        expect(function() { testFunc(null, [], null, null, null, null) }).toThrow('Wrong type for parameter "obj" of testFunc: Expected Object, but got Array.');
    });
    it('should throw when given invalid array', function() {
        var testFunc = createTestFunc(true);
        expect(function() { testFunc(null, null, {}, null, null, null) }).toThrow('Wrong type for parameter "arr" of testFunc: Expected Array, but got Object.');
    });
    it('should throw when given invalid string', function() {
        var testFunc = createTestFunc(true);
        expect(function() { testFunc(null, null, null, 5, null, null) }).toThrow('Wrong type for parameter "str" of testFunc: Expected String, but got Number.');
    });
    it('should throw when given invalid date', function() {
        var testFunc = createTestFunc(true);
        expect(function() { testFunc(null, null, null, null, 233, null) }).toThrow('Wrong type for parameter "date" of testFunc: Expected Date, but got Number.');
    });
    it('should throw when given invalid function', function() {
        var testFunc = createTestFunc(true);
        expect(function() { testFunc(null, null, null, null, null, new Date) }).toThrow('Wrong type for parameter "func" of testFunc: Expected Function, but got Date.');
    });
    it('should not throw when checking is disabled', function() {
        var testFunc = createTestFunc(false);
        argscheck.enableChecks = false;
        testFunc();
    });
});
