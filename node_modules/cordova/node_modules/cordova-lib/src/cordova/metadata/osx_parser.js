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

/* jshint sub:true */

var fs            = require('fs'),
    unorm         = require('unorm'),
    path          = require('path'),
    xcode         = require('xcode'),
    util          = require('../util'),
    events        = require('cordova-common').events,
    shell         = require('shelljs'),
    plist         = require('plist'),
    Q             = require('q'),
    Parser        = require('./parser'),
    ios_parser    = require('./ios_parser'),
    ConfigParser = require('cordova-common').ConfigParser,
    CordovaError = require('cordova-common').CordovaError;

function osx_parser(project) {

    try {
        var xcodeproj_dir = fs.readdirSync(project).filter(function(e) { return e.match(/\.xcodeproj$/i); })[0];
        if (!xcodeproj_dir) throw new CordovaError('The provided path "' + project + '" is not a Cordova OS X project.');

        // Call the base class constructor
        Parser.call(this, 'osx', project);

        this.xcodeproj = path.join(project, xcodeproj_dir);
        this.originalName = this.xcodeproj.substring(this.xcodeproj.lastIndexOf(path.sep)+1, this.xcodeproj.indexOf('.xcodeproj'));
        this.cordovaproj = path.join(project, this.originalName);
    } catch(e) {
        console.error(e);
        throw new CordovaError('The provided path "'+project+'" is not a Cordova OS X project.');
    }
    this.path = unorm.nfd(project);
    this.pbxproj = path.join(this.xcodeproj, 'project.pbxproj');
    this.config_path = path.join(this.cordovaproj, 'config.xml');
}

require('util').inherits(osx_parser, ios_parser);

module.exports = osx_parser;

// Returns a promise.
osx_parser.prototype.update_from_config = function(config) {
    if (config instanceof ConfigParser) {
    } else {
        return Q.reject(new Error('update_from_config requires a ConfigParser object'));
    }
    // CB-6992 it is necessary to normalize characters
    // because node and shell scripts handles unicode symbols differently
    // We need to normalize the name to NFD form since iOS uses NFD unicode form
    var name = unorm.nfd(config.name());
    var pkg = config.ios_CFBundleIdentifier() || config.packageName();
    var version = config.version();

    // Update package id (bundle id)
    var plistFile = path.join(this.cordovaproj, this.originalName + '-Info.plist');
    var infoPlist = plist.parse(fs.readFileSync(plistFile, 'utf8'));
    infoPlist['CFBundleIdentifier'] = pkg;

    // Update version (bundle version)
    infoPlist['CFBundleShortVersionString'] = version;
    var CFBundleVersion = config.ios_CFBundleVersion() || default_CFBundleVersion(version);
    infoPlist['CFBundleVersion'] = CFBundleVersion;

    var info_contents = plist.build(infoPlist);
    info_contents = info_contents.replace(/<string>[\s\r\n]*<\/string>/g,'<string></string>');
    fs.writeFileSync(plistFile, info_contents, 'utf-8');
    events.emit('verbose', 'Wrote out OS X Bundle Identifier to "' + pkg + '"');
    events.emit('verbose', 'Wrote out OS X Bundle Version to "' + version + '"');

    // Update icons
    var icons = config.getIcons('osx');
    var platformRoot = this.cordovaproj;
    var appRoot = util.isCordova(platformRoot);

    // See https://developer.apple.com/library/mac/documentation/UserExperience/Conceptual/OSXHIGuidelines/Designing.html
    // for application images sizes reference.
    var platformIcons = [
        {dest: 'icon-512x512.png', width: 512, height: 512},
        {dest: 'icon-256x256.png', width: 256, height: 256},
        {dest: 'icon-128x128.png', width: 128, height: 128},
        {dest: 'icon-64x64.png', width: 64, height: 64},
        {dest: 'icon-32x32.png', width: 32, height: 32},
        {dest: 'icon-16x16.png', width: 16, height: 16}
    ];

    platformIcons.forEach(function (item) {
        var icon = icons.getBySize(item.width, item.height) || icons.getDefault();
        if (icon){
            var src = path.join(appRoot, icon.src),
                dest = path.join(platformRoot, 'Images.xcassets/AppIcon.appiconset/', item.dest);
            events.emit('verbose', 'Copying icon from ' + src + ' to ' + dest);
            shell.cp('-f', src, dest);
        }
    });

    var parser = this;
    return this.update_build_settings(config).then(function() {
        if (name == parser.originalName) {
            events.emit('verbose', 'OS X Product Name has not changed (still "' + parser.originalName + '")');
            return Q();
        }

        // Update product name inside pbxproj file
        var proj = new xcode.project(parser.pbxproj);
        try {
            proj.parseSync();
        } catch (err) {
            return Q.reject(new Error('An error occured during parsing of project.pbxproj. Start weeping. Output: ' + err));
        }

        proj.updateProductName(name);
        fs.writeFileSync(parser.pbxproj, proj.writeSync(), 'utf-8');

        // Move the xcodeproj and other name-based dirs over.
        shell.mv(path.join(parser.cordovaproj, parser.originalName + '-Info.plist'), path.join(parser.cordovaproj, name + '-Info.plist'));
        shell.mv(path.join(parser.cordovaproj, parser.originalName + '-Prefix.pch'), path.join(parser.cordovaproj, name + '-Prefix.pch'));
        // CB-8914 remove userdata otherwise project is un-usable in xcode
        shell.rm('-rf',path.join(parser.xcodeproj,'xcuserdata/'));
        shell.mv(parser.xcodeproj, path.join(parser.path, name + '.xcodeproj'));
        shell.mv(parser.cordovaproj, path.join(parser.path, name));

        // Update self object with new paths
        var old_name = parser.originalName;
        parser = new module.exports(parser.path);

        // Hack this shi*t
        var pbx_contents = fs.readFileSync(parser.pbxproj, 'utf-8');
        pbx_contents = pbx_contents.split(old_name).join(name);
        fs.writeFileSync(parser.pbxproj, pbx_contents, 'utf-8');
        events.emit('verbose', 'Wrote out OS X Product Name and updated XCode project file names from "'+old_name+'" to "' + name + '".');

        return Q();
    });
};

// Construct a default value for CFBundleVersion as the version with any
// -rclabel stripped=.
function default_CFBundleVersion(version) {
    return version.split('-')[0];
}