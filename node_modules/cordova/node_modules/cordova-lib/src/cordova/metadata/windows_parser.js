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
    path          = require('path'),
    util          = require('../util'),
    events        = require('cordova-common').events,
    shell         = require('shelljs'),
    Q             = require('q'),
    Parser        = require('./parser'),
    ConfigParser = require('cordova-common').ConfigParser,
    CordovaError = require('cordova-common').CordovaError,
    xml           = require('cordova-common').xmlHelpers,
    HooksRunner        = require('../../hooks/HooksRunner');

function windows_parser(project) {

    try {
        this.isOldProjectTemplate = false;
        // Check that it's a universal windows store project
        var projFile = fs.readdirSync(project).filter(function(e) { return e.match(/\.projitems$/i); })[0];
        if (!projFile) {
            this.isOldProjectTemplate = true;
            projFile = fs.readdirSync(project).filter(function(e) { return e.match(/\.jsproj$/i); })[0];
        }
        if (!projFile) {
            throw new CordovaError('No project file in "'+project+'"');
        }

        // Call the base class constructor
        Parser.call(this, 'windows8', project);

        this.projDir = project;
        this.projFilePath = path.join(this.projDir, projFile);

        if (this.isOldProjectTemplate) {
            this.manifestPath = path.join(this.projDir, 'package.appxmanifest');
        }

    } catch(e) {
        throw new CordovaError('The provided path "' + project + '" is not a Windows project. ' + e);
    }
}

require('util').inherits(windows_parser, Parser);

module.exports = windows_parser;

windows_parser.prototype.update_from_config = function(config) {

    //check config parser
    if (config instanceof ConfigParser) {
    } else throw new Error('update_from_config requires a ConfigParser object');

    if (!this.isOldProjectTemplate) {
        // If there is platform-defined prepare script, require and exec it
        var platformPrepare = require(path.join(this.projDir, 'cordova', 'lib', 'prepare'));
        platformPrepare.applyPlatformConfig();
        return;
    }

    // code below is required for compatibility reason. New template version is not required this anymore.

    //Get manifest file
    var manifest = xml.parseElementtreeSync(this.manifestPath);

    var version = this.fixConfigVersion(config.version());
    var name = config.name();
    var pkgName = config.packageName();
    var author = config.author();

    var identityNode = manifest.find('.//Identity');
    if(identityNode) {
        // Update app name in identity
        var appIdName = identityNode['attrib']['Name'];
        if (appIdName != pkgName) {
            identityNode['attrib']['Name'] = pkgName;
        }

        // Update app version
        var appVersion = identityNode['attrib']['Version'];
        if(appVersion != version) {
            identityNode['attrib']['Version'] = version;
        }
    }

    // Update name (windows8 has it in the Application[@Id] and Application.VisualElements[@DisplayName])
    var app = manifest.find('.//Application');
    if(app) {

        var appId = app['attrib']['Id'];

        if (appId != pkgName) {
            app['attrib']['Id'] = pkgName;
        }

        var visualElems = manifest.find('.//VisualElements') || manifest.find('.//m2:VisualElements');

        if(visualElems) {
            var displayName = visualElems['attrib']['DisplayName'];
            if(displayName != name) {
                visualElems['attrib']['DisplayName'] = name;
            }
        }
        else {
            throw new Error('update_from_config expected a valid package.appxmanifest' +
                            ' with a <VisualElements> node');
        }
    }
    else {
        throw new Error('update_from_config expected a valid package.appxmanifest' +
                        ' with a <Application> node');
    }

    // Update properties
    var properties = manifest.find('.//Properties');
    if (properties && properties.find) {
        var displayNameElement = properties.find('.//DisplayName');
        if (displayNameElement && displayNameElement.text != name) {
            displayNameElement.text = name;
        }

        var publisherNameElement = properties.find('.//PublisherDisplayName');
        if (publisherNameElement && publisherNameElement.text != author) {
            publisherNameElement.text = author;
        }
    }

    // sort Capability elements as per CB-5350 Windows8 build fails due to invalid 'Capabilities' definition
    // to sort elements we remove them and then add again in the appropriate order
    var capabilitiesRoot = manifest.find('.//Capabilities'),
        capabilities = capabilitiesRoot._children || [];

    capabilities.forEach(function(elem){
        capabilitiesRoot.remove(elem);
    });
    capabilities.sort(function(a, b) {
        return (a.tag > b.tag)? 1: -1;
    });
    capabilities.forEach(function(elem){
        capabilitiesRoot.append(elem);
    });

    //Write out manifest
    fs.writeFileSync(this.manifestPath, manifest.write({indent: 4}), 'utf-8');

    // Update icons
    var icons = config.getIcons('windows8');
    var platformRoot = this.projDir;
    var appRoot = util.isCordova(platformRoot);

    // Icons, that should be added to platform
    var platformIcons = [
        {dest: 'images/logo.png', width: 150, height: 150},
        {dest: 'images/smalllogo.png', width: 30, height: 30},
        {dest: 'images/storelogo.png', width: 50, height: 50},
    ];

    platformIcons.forEach(function (item) {
        var icon = icons.getBySize(item.width, item.height) || icons.getDefault();
        if (icon){
            var src = path.join(appRoot, icon.src),
                dest = path.join(platformRoot, item.dest);
            events.emit('verbose', 'Copying icon from ' + src + ' to ' + dest);
            shell.cp('-f', src, dest);
        }
    });

    // Update splashscreen
    // Image size for Windows 8 should be 620 x 300 px
    // See http://msdn.microsoft.com/en-us/library/windows/apps/hh465338.aspx for reference
    var splash = config.getSplashScreens('windows8').getBySize(620, 300);
    if (splash){
        var src = path.join(appRoot, splash.src),
            dest = path.join(platformRoot, 'images/splashscreen.png');
        events.emit('verbose', 'Copying icon from ' + src + ' to ' + dest);
        shell.cp('-f', src, dest);
    }
};

// Returns the platform-specific www directory.
windows_parser.prototype.www_dir = function() {
    return path.join(this.projDir, 'www');
};

windows_parser.prototype.config_xml = function() {
    return path.join(this.projDir,'config.xml');
};

// copy files from merges directory to actual www dir
windows_parser.prototype.copy_merges = function(merges_sub_path) {
    var merges_path = path.join(util.appDir(util.isCordova(this.projDir)), 'merges', merges_sub_path);
    if (fs.existsSync(merges_path)) {
        var overrides = path.join(merges_path, '*');
        shell.cp('-rf', overrides, this.www_dir());
    }
};

// Used for creating platform_www in projects created by older versions.
windows_parser.prototype.cordovajs_path = function(libDir) {
    var jsPath = path.join(libDir, 'template', 'www', 'cordova.js');
    return path.resolve(jsPath);
};

windows_parser.prototype.cordovajs_src_path = function(libDir) {
    var jsPath = path.join(libDir, 'cordova-js-src');
    return path.resolve(jsPath);
};

// Replace the www dir with contents of platform_www and app www and updates the csproj file.
windows_parser.prototype.update_www = function() {
    var projectRoot = util.isCordova(this.projDir);
    var app_www = util.projectWww(projectRoot);
    var platform_www = path.join(this.projDir, 'platform_www');

    // Clear the www dir
    shell.rm('-rf', this.www_dir());
    shell.mkdir(this.www_dir());
    // Copy over all app www assets
    shell.cp('-rf', path.join(app_www, '*'), this.www_dir());

    // Copy all files from merges directory.
    // CB-6976 Windows Universal Apps. For smooth transition from 'windows8' platform
    // we allow using 'windows8' merges for new 'windows' platform
    this.copy_merges('windows8');
    this.copy_merges('windows');

    // Copy over stock platform www assets (cordova.js)
    shell.cp('-rf', path.join(platform_www, '*'), this.www_dir());
};

// calls the nessesary functions to update the windows8 project
windows_parser.prototype.update_project = function(cfg) {
    // console.log("Updating windows8 project...");

    try {
        this.update_from_config(cfg);
    } catch(e) {
        return Q.reject(e);
    }

    var that = this;
    var projectRoot = util.isCordova(process.cwd());

    var hooksRunner = new HooksRunner(projectRoot);
    return hooksRunner.fire('pre_package', { wwwPath:this.www_dir(), platforms: [this.isOldProjectTemplate ? 'windows8' : 'windows'] })
    .then(function() {
        // overrides (merges) are handled in update_www()
        that.add_bom();
        util.deleteSvnFolders(that.www_dir());
    });
};

// Adjust version number as per CB-5337 Windows8 build fails due to invalid app version
windows_parser.prototype.fixConfigVersion = function (version) {
    if(version && version.match(/\.\d/g)) {
        var numVersionComponents = version.match(/\.\d/g).length + 1;
        while (numVersionComponents++ < 4) {
            version += '.0';
        }
    }
    return version;
};

// CB-5421 Add BOM to all html, js, css files to ensure app can pass Windows Store Certification
windows_parser.prototype.add_bom = function () {
    var www = this.www_dir();
    var files = shell.ls('-R', www);

    files.forEach(function (file) {
        if (!file.match(/\.(js|html|css|json)$/i)) {
            return;
        }

        var filePath = path.join(www, file);
        // skip if this is a folder
        if (!fs.lstatSync(filePath).isFile()) {
            return;
        }

        var content = fs.readFileSync(filePath);

        if (content[0] !== 0xEF && content[1] !== 0xBE && content[2] !== 0xBB) {
            fs.writeFileSync(filePath, '\ufeff' + content);
        }
    });
};
