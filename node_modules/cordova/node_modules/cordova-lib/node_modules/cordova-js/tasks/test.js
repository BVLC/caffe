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
*/
try {
    require('jasmine-node');
} catch (e) {
    console.error("\njasmine-node is not installed, you need to:\n\trun `npm install` from " + require('path').dirname(__dirname)+"\n");
    process.exit(1);
}

module.exports = function(grunt) {
    grunt.registerTask('_test', 'Runs test in node', function() {
        var done = this.async();
        require('./lib/test-jsdom')(done);
    });

    grunt.registerTask('_btest', 'Runs tests in the browser', function() {
        require('./lib/test-browser')();
        this.async(); // never finish.
    });
};
