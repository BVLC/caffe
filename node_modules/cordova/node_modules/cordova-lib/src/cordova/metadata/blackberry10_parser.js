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

var fs            = require('fs'),
    path          = require('path'),
    shell         = require('shelljs'),
    util          = require('../util'),
    Q             = require('q'),
    Parser        = require('./parser'),
    ConfigParser = require('cordova-common').ConfigParser,
    CordovaError = require('cordova-common').CordovaError,
    events = require('cordova-common').events;

function blackberry_parser(project) {
    if (!fs.existsSync(path.join(project, 'www'))) {
        throw new CordovaError('The provided path "' + project + '" is not a Cordova BlackBerry10 project.');
    }

    // Call the base class constructor
    Parser.call(this, 'blackberry10', project);

    this.path = project;
    this.config_path = path.join(this.path, 'www', 'config.xml');
    this.xml = new ConfigParser(this.config_path);
}

require('util').inherits(blackberry_parser, Parser);

module.exports = blackberry_parser;

blackberry_parser.prototype.update_from_config = function(config) {
    var projectRoot = util.isCordova(this.path),
        resDir = path.join(this.path, 'platform_www', 'res'),
        icons,
        i;

    if (!config instanceof ConfigParser) {
        throw new Error('update_from_config requires a ConfigParser object');
    }

    shell.rm('-rf', resDir);
    shell.mkdir(resDir);

    icons = config.getIcons('blackberry10');
    if (icons) {
        for (i = 0; i < icons.length; i++) {
            var src = path.join(projectRoot, icons[i].src),
                dest = path.join(this.path, 'platform_www', icons[i].src),
                destFolder = path.dirname(dest);

            if (!fs.existsSync(destFolder)) {
                shell.mkdir('-p', destFolder); // make sure target dir exists
            }
            events.emit('verbose', 'Copying icon from ' + src + ' to ' + dest);
            shell.cp('-f', src, dest);
        }
    }
};

// Returns a promise.
blackberry_parser.prototype.update_project = function(cfg) {
    var self = this;

    try {
        self.update_from_config(cfg);
    } catch(e) {
        return Q.reject(e);
    }
    self.update_overrides();
    util.deleteSvnFolders(this.www_dir());
    return Q();
};

// Returns the platform-specific www directory.
blackberry_parser.prototype.www_dir = function() {
    return path.join(this.path, 'www');
};

blackberry_parser.prototype.config_xml = function(){
    return this.config_path;
};

// Used for creating platform_www in projects created by older versions.
blackberry_parser.prototype.cordovajs_path = function(libDir) {
    var jsPath = path.join(libDir, 'javascript', 'cordova.blackberry10.js');
    return path.resolve(jsPath);
};

blackberry_parser.prototype.cordovajs_src_path = function(libDir) {
    var jsPath = path.join(libDir, 'cordova-js-src');
    return path.resolve(jsPath);
};

// Replace the www dir with contents of platform_www and app www.
blackberry_parser.prototype.update_www = function() {
    var projectRoot = util.isCordova(this.path);
    var app_www = util.projectWww(projectRoot);
    var platform_www = path.join(this.path, 'platform_www');
    var platform_cfg_backup = new ConfigParser(this.config_path);

    // Clear the www dir
    shell.rm('-rf', this.www_dir());
    shell.mkdir(this.www_dir());
    // Copy over all app www assets
    shell.cp('-rf', path.join(app_www, '*'), this.www_dir());
    // Copy over stock platform www assets (cordova.js)
    shell.cp('-rf', path.join(platform_www, '*'), this.www_dir());
    //Re-Write config.xml
    platform_cfg_backup.write();
};

// update the overrides folder into the www folder
blackberry_parser.prototype.update_overrides = function() {
    var projectRoot = util.isCordova(this.path);
    var merges_path = path.join(util.appDir(projectRoot), 'merges', 'blackberry10');
    if (fs.existsSync(merges_path)) {
        var overrides = path.join(merges_path, '*');
        shell.cp('-rf', overrides, this.www_dir());
    }
};
