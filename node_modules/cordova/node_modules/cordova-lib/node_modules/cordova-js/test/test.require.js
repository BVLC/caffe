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

describe("require + define", function () {
    it("exists off of cordova", function () {
        var cordova = require('cordova');
        expect(cordova.require).toBeDefined();
        expect(cordova.define).toBeDefined();
    });

    describe("when defining", function () {
        it("can define and remove module", function () {
            define("a", jasmine.createSpy());
            define.remove("a");
        });

        it("can remove a module that doesn't exist", function () {
            define.remove("can't touch this");
        });

        it("throws an error the module already exists", function () {
            expect(function () {
                define("cordova", function () {});
            }).toThrow("module cordova already defined");
        });

        it("doesn't call the factory method when defining", function () {
            var factory = jasmine.createSpy();
            define("ff", factory);
            expect(factory).not.toHaveBeenCalled();
        });
    });

    describe("when requiring", function () {
        it("throws an exception when module doesn't exist", function () {
            expect(function () {
                require("your mom");
            }).toThrow("module your mom not found");
        });

        it("throws an exception when modules depend on each other", function () {
            define("ModuleA", function(require, exports, module) {
                require("ModuleB");
            });
            define("ModuleB", function(require, exports, module) {
                require("ModuleA");
            });
            expect(function () {
                require("ModuleA");
            }).toThrow("Cycle in require graph: ModuleA->ModuleB->ModuleA");
            define.remove("ModuleA");
            define.remove("ModuleB");
        });

        it("throws an exception when a cycle of requires occurs", function () {
            define("ModuleA", function(require, exports, module) {
                require("ModuleB");
            });
            define("ModuleB", function(require, exports, module) {
                require("ModuleC");
            });
            define("ModuleC", function(require, exports, module) {
                require("ModuleA");
            });
            expect(function () {
                require("ModuleA");
            }).toThrow("Cycle in require graph: ModuleA->ModuleB->ModuleC->ModuleA");
            define.remove("ModuleA");
            define.remove("ModuleB");
            define.remove("ModuleC");
        });

        it("calls the factory method when requiring", function () {
            var factory = jasmine.createSpy();
            define("dino", factory);
            require("dino");

            expect(factory).toHaveBeenCalledWith(jasmine.any(Function),
                {}, {
                    id: "dino",
                    exports: {}
                });

            define.remove("dino");
        });

        it("returns the exports object", function () {
            define("a", function (require, exports, module) {
                exports.stuff = "asdf";
            });

            var v = require("a");
            expect(v.stuff).toBe("asdf");
            define.remove("a");
        });

        it("can use both the exports and module.exports object", function () {
            define("a", function (require, exports, module) {
                exports.a = "a";
                module.exports.b = "b";
            });

            var v = require("a");
            expect(v.a).toBe("a");
            expect(v.b).toBe("b");
            define.remove("a");
        });

        it("returns was is assigned to module.exports", function () {
            var Foo = function () { };
            define("a", function (require, exports, module) {
                module.exports = new Foo();
            });

            var v = require("a");
            expect(v instanceof Foo).toBe(true);
            define.remove("a");
        });

        it("has the id and exports values but not the factory on the module object", function () {
            var factory = function (require, exports, module) {
                expect(module.id).toBe("a");
                expect(module.exports).toBeDefined();
                expect(module.factory).not.toBeDefined();
            };

            define("a", factory);
            require("a");
        });
    });
});
