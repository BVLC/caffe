/**
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
*/

var validateIdentifier = require('../valid-identifier');

describe('valid-identifier',function(done){

	it('should allow valid identifiers', function(done) {
		expect(validateIdentifier("LeadingCap")).toBe(true);
		expect(validateIdentifier("camelCase")).toBe(true);
		expect(validateIdentifier("ALLCAPS")).toBe(true);
		expect(validateIdentifier("x")).toBe(true);
		expect(validateIdentifier("_underscore")).toBe(true);
		expect(validateIdentifier("$dollarSign")).toBe(true);
		expect(validateIdentifier("_8bit")).toBe(true);
		expect(validateIdentifier("LeadingCap.camelCase.ALLCAPS._underscore.$dollarSign")).toBe(true);
		done();
	});

	it('should not allow invalid identifiers', function(done) {
		expect(validateIdentifier("3numberstart")).toBe(false);
		expect(validateIdentifier("has space")).toBe(false);
		expect(validateIdentifier("x + y")).toBe(false);
		expect(validateIdentifier("O'Mega")).toBe(false);
		expect(validateIdentifier("amp&ersand")).toBe(false);
		done();
	});

	it('should not allow reserved words', function(done) {
		expect(validateIdentifier("foreach")).toBe(false);
		expect(validateIdentifier("valid.foreach")).toBe(false);
		expect(validateIdentifier("win.in.g")).toBe(false);
		expect(validateIdentifier("ends.with.")).toBe(false);
		expect(validateIdentifier(".starts.with")).toBe(false);
		expect(validateIdentifier("double..dot")).toBe(false);
		done();
	});

});  