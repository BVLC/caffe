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
module.exports = function(grunt) {

    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),
        compile: {
            "amazon-fireos": {},
            "android": {},
            "blackberry10": {},
            "ios": {},
            "osx": {},
            "test": {},
            "windows": { useWindowsLineEndings: true },
            "wp8": { useWindowsLineEndings: true },
            "firefoxos": {},
            "webos": {},
            "ubuntu": {},
            "browser": {}
        },
        "compile-browserify": {
            "amazon-fireos": {},
            "android": {},
            "blackberry10": {},
            "ios": {},
            "osx": {},
            "test": {},
            "windows": { useWindowsLineEndings: true },
            "wp8": { useWindowsLineEndings: true },
            "firefoxos": {},
            "webos": {},
            "ubuntu": {},
            "browser": {}
        },
        clean: ['pkg'],
        jshint: {
            options: {
                jshintrc: '.jshintrc',
            },
            src: ['src/**/*.js']
        },
    });

    // external tasks
    grunt.loadNpmTasks('grunt-contrib-clean');
    grunt.loadNpmTasks('grunt-contrib-jshint');

    // custom tasks
    grunt.loadTasks('tasks');

    // defaults
    grunt.registerTask('default', ['build', 'test']);
    grunt.registerTask('build', ['compile', 'jshint', 'whitespace-check']);
    grunt.registerTask('test', ['compile:test', 'jshint', '_test']);
    grunt.registerTask('btest', ['compile:test', 'jshint', '_btest']);
    grunt.registerTask('cover', ['compile', '_cover']);
    grunt.registerTask('test-browserify', ['compile-browserify:test', 'jshint', '_test']);
    grunt.registerTask('btest-browserify', ['compile-browserify:test', 'jshint', '_btest']);
};
