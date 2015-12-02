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

describe("builder", function () {

    var builder = require('cordova/builder');

    it("includes the module into the target", function () {

        var target = {},
            objects = {
                foo: {
                    path: "cordova/builder"
                }
            };


        builder.buildIntoAndClobber(objects, target);
        expect(target.foo).toBeDefined();
        expect(target.foo).toBe(require("cordova/builder"));
    });

    it("returns an empty object literal if no path", function () {
        var target = {},
            objects = {cat: {}};

        builder.buildIntoButDoNotClobber(objects, target);

        expect(target.cat).toBeDefined();
    });

    it("builds out the children", function () {

        var target = {},
            objects = {
               homer: {
                   children: {
                       bart: {},
                       lisa: {},
                       maggie: {
                           path: "cordova/builder"
                       }
                   }
               }
            };

        builder.buildIntoButDoNotClobber(objects, target);

        expect(target.homer.bart).toBeDefined();
        expect(target.homer.maggie).toBe(require('cordova/builder'));
        expect(target.homer.lisa).toBeDefined();
    });
});
