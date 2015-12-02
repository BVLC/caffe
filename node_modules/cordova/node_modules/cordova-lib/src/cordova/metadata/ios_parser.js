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
    ConfigParser = require('cordova-common').ConfigParser,
    URL           = require('url'),
    CordovaError = require('cordova-common').CordovaError;

function ios_parser(project) {

    try {
        var xcodeproj_dir = fs.readdirSync(project).filter(function(e) { return e.match(/\.xcodeproj$/i); })[0];
        if (!xcodeproj_dir) throw new CordovaError('The provided path "' + project + '" is not a Cordova iOS project.');

        // Call the base class constructor
        Parser.call(this, 'ios', project);

        this.xcodeproj = path.join(project, xcodeproj_dir);
        this.originalName = this.xcodeproj.substring(this.xcodeproj.lastIndexOf(path.sep)+1, this.xcodeproj.indexOf('.xcodeproj'));
        this.cordovaproj = path.join(project, this.originalName);
    } catch(e) {
        throw new CordovaError('The provided path "'+project+'" is not a Cordova iOS project.');
    }
    this.path = unorm.nfd(project);
    this.pbxproj = path.join(this.xcodeproj, 'project.pbxproj');
    this.config_path = path.join(this.cordovaproj, 'config.xml');
}

require('util').inherits(ios_parser, Parser);

module.exports = ios_parser;

// Returns a promise.
ios_parser.prototype.update_from_config = function(config) {
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

    var orientation = this.helper.getOrientation(config);

    if (orientation && !this.helper.isDefaultOrientation(orientation)) {
        switch (orientation.toLowerCase()) {
            case 'portrait':
                infoPlist['UIInterfaceOrientation'] = [ 'UIInterfaceOrientationPortrait' ];
                infoPlist['UISupportedInterfaceOrientations'] = [ 'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationPortraitUpsideDown' ];
                infoPlist['UISupportedInterfaceOrientations~ipad'] = [ 'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationPortraitUpsideDown' ];
                break;
            case 'landscape':
                infoPlist['UIInterfaceOrientation'] = [ 'UIInterfaceOrientationLandscapeLeft' ];
                infoPlist['UISupportedInterfaceOrientations'] = [ 'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight' ];
                infoPlist['UISupportedInterfaceOrientations~ipad'] = [ 'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight' ];
                break;
            case 'all':
                infoPlist['UIInterfaceOrientation'] = [ 'UIInterfaceOrientationPortrait' ];
                infoPlist['UISupportedInterfaceOrientations'] = [ 'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationPortraitUpsideDown', 'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight' ];
                infoPlist['UISupportedInterfaceOrientations~ipad'] = [ 'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationPortraitUpsideDown', 'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight' ];
                break;
            default:
                infoPlist['UIInterfaceOrientation'] = [ orientation ];
                delete infoPlist['UISupportedInterfaceOrientations'];
                delete infoPlist['UISupportedInterfaceOrientations~ipad'];
        }
    } else {
        delete infoPlist['UISupportedInterfaceOrientations'];
        delete infoPlist['UISupportedInterfaceOrientations~ipad'];
        delete infoPlist['UIInterfaceOrientation'];
    }
    
    var ats = (infoPlist['NSAppTransportSecurity'] || {});
    ats = processAccessEntriesAsATS(config, ats);
    ats = processAllowNavigationEntriesAsATS(config, ats);
    infoPlist['NSAppTransportSecurity'] = ats;

    var info_contents = plist.build(infoPlist);
    info_contents = info_contents.replace(/<string>[\s\r\n]*<\/string>/g,'<string></string>');
    fs.writeFileSync(plistFile, info_contents, 'utf-8');
    events.emit('verbose', 'Wrote out iOS Bundle Identifier to "' + pkg + '"');
    events.emit('verbose', 'Wrote out iOS Bundle Version to "' + version + '"');

    // Update icons
    var icons = config.getIcons('ios');
    var platformRoot = this.cordovaproj;
    var appRoot = util.isCordova(platformRoot);

    // See https://developer.apple.com/library/ios/documentation/userexperience/conceptual/mobilehig/LaunchImages.html
    // for launch images sizes reference.
    var platformIcons = [
        {dest: 'icon-60.png', width: 60, height: 60},
        {dest: 'icon-60@2x.png', width: 120, height: 120},
        {dest: 'icon-60@3x.png', width: 180, height: 180},
        {dest: 'icon-76.png', width: 76, height: 76},
        {dest: 'icon-76@2x.png', width: 152, height: 152},
        {dest: 'icon-small.png', width: 29, height: 29},
        {dest: 'icon-small@2x.png', width: 58, height: 58},
        {dest: 'icon-40.png', width: 40, height: 40},
        {dest: 'icon-40@2x.png', width: 80, height: 80},
        {dest: 'icon.png', width: 57, height: 57},
        {dest: 'icon@2x.png', width: 114, height: 114},
        {dest: 'icon-72.png', width: 72, height: 72},
        {dest: 'icon-72@2x.png', width: 144, height: 144},
        {dest: 'icon-50.png', width: 50, height: 50},
        {dest: 'icon-50@2x.png', width: 100, height: 100}
    ];
    
    var destIconsFolder, destSplashFolder;
    var xcassetsExists = folderExists(path.join(platformRoot, 'Images.xcassets/'));
    
    if (xcassetsExists) {
        destIconsFolder = 'Images.xcassets/AppIcon.appiconset/';
    } else {
        destIconsFolder = 'Resources/icons/';
    }

    platformIcons.forEach(function (item) {
        var icon = icons.getBySize(item.width, item.height) || icons.getDefault();
        if (icon){
            var src = path.join(appRoot, icon.src),
                dest = path.join(platformRoot, destIconsFolder, item.dest);
            events.emit('verbose', 'Copying icon from ' + src + ' to ' + dest);
            shell.cp('-f', src, dest);
        }
    });

    // Update splashscreens
    var splashScreens = config.getSplashScreens('ios');
    var platformSplashScreens = [
        {dest: 'Default~iphone.png', width: 320, height: 480},
        {dest: 'Default@2x~iphone.png', width: 640, height: 960},
        {dest: 'Default-Portrait~ipad.png', width: 768, height: 1024},
        {dest: 'Default-Portrait@2x~ipad.png', width: 1536, height: 2048},
        {dest: 'Default-Landscape~ipad.png', width: 1024, height: 768},
        {dest: 'Default-Landscape@2x~ipad.png', width: 2048, height: 1536},
        {dest: 'Default-568h@2x~iphone.png', width: 640, height: 1136},
        {dest: 'Default-667h.png', width: 750, height: 1334},
        {dest: 'Default-736h.png', width: 1242, height: 2208},
        {dest: 'Default-Landscape-736h.png', width: 2208, height: 1242}
    ];
    
    if (xcassetsExists) {
        destSplashFolder = 'Images.xcassets/LaunchImage.launchimage/';
    } else {
        destSplashFolder = 'Resources/splash/';
    }

    platformSplashScreens.forEach(function(item) {
        var splash = splashScreens.getBySize(item.width, item.height);
        if (splash){
            var src = path.join(appRoot, splash.src),
                dest = path.join(platformRoot, destSplashFolder, item.dest);
            events.emit('verbose', 'Copying splash from ' + src + ' to ' + dest);
            shell.cp('-f', src, dest);
        }
    });

    var parser = this;
    return this.update_build_settings(config).then(function() {
        if (name == parser.originalName) {
            events.emit('verbose', 'iOS Product Name has not changed (still "' + parser.originalName + '")');
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
        events.emit('verbose', 'Wrote out iOS Product Name and updated XCode project file names from "'+old_name+'" to "' + name + '".');

        return Q();
    });
};

// Returns the platform-specific www directory.
ios_parser.prototype.www_dir = function() {
    return path.join(this.path, 'www');
};

ios_parser.prototype.config_xml = function(){
    return this.config_path;
};

// Used for creating platform_www in projects created by older versions.
ios_parser.prototype.cordovajs_path = function(libDir) {
    var jsPath = path.join(libDir, 'CordovaLib', 'cordova.js');
    return path.resolve(jsPath);
};

ios_parser.prototype.cordovajs_src_path = function(libDir) {
    var jsPath = path.join(libDir, 'cordova-js-src');
    return path.resolve(jsPath);
};

// Replace the www dir with contents of platform_www and app www.
ios_parser.prototype.update_www = function() {
    var projectRoot = util.isCordova(this.path);
    var app_www = util.projectWww(projectRoot);
    var platform_www = path.join(this.path, 'platform_www');

    // Clear the www dir
    shell.rm('-rf', this.www_dir());
    shell.mkdir(this.www_dir());
    // Copy over all app www assets
    shell.cp('-rf', path.join(app_www, '*'), this.www_dir());
    // Copy over stock platform www assets (cordova.js)
    shell.cp('-rf', path.join(platform_www, '*'), this.www_dir());
};

// update the overrides folder into the www folder
ios_parser.prototype.update_overrides = function() {
    var projectRoot = util.isCordova(this.path);
    var merges_path = path.join(util.appDir(projectRoot), 'merges', 'ios');
    if (fs.existsSync(merges_path)) {
        var overrides = path.join(merges_path, '*');
        shell.cp('-rf', overrides, this.www_dir());
    }
};

// Returns a promise.
ios_parser.prototype.update_project = function(cfg) {
    var self = this;
    return this.update_from_config(cfg)
    .then(function() {
        self.update_overrides();
        util.deleteSvnFolders(self.www_dir());
    });
};

ios_parser.prototype.update_build_settings = function(config) {
    var targetDevice = parseTargetDevicePreference(config.getPreference('target-device', 'ios'));
    var deploymentTarget = config.getPreference('deployment-target', 'ios');

    // no build settings provided, we don't need to parse and update .pbxproj file
    if (!targetDevice && !deploymentTarget) {
        return Q();
    }

    var proj = new xcode.project(this.pbxproj);

    try {
        proj.parseSync();
    } catch (err) {
        return Q.reject(new Error('An error occured during parsing of project.pbxproj. Start weeping. Output: ' + err));
    }

    if (targetDevice) {
        events.emit('verbose', 'Set TARGETED_DEVICE_FAMILY to ' + targetDevice + '.');
        proj.updateBuildProperty('TARGETED_DEVICE_FAMILY', targetDevice);
    }

    if (deploymentTarget) {
        events.emit('verbose', 'Set IPHONEOS_DEPLOYMENT_TARGET to "' + deploymentTarget + '".');
        proj.updateBuildProperty('IPHONEOS_DEPLOYMENT_TARGET', deploymentTarget);
    }

    fs.writeFileSync(this.pbxproj, proj.writeSync(), 'utf-8');

    return Q();
};

function processAccessEntriesAsATS(config, ats) {
    // CB-9569 - Support <access> tag to Application Transport Security (ATS) in iOS 9+
    var accesses = config.getAccesses();
    
    // the default is the wildcard, so if there are no access tags, we add the wildcard in
    if (accesses.length === 0) {
        accesses.push({ 'origin' : '*'});
    }
    
    if (!ats) {
        ats = {};
    }
    
    accesses.forEach(function(access) {
        ats = processUrlAsATS(ats, access.origin, access.minimum_tls_version, access.requires_forward_secrecy);
    });
    
    return ats;
}

function processAllowNavigationEntriesAsATS(config, ats) {
    // CB-9569 - Support <allow-navigation> tag to Application Transport Security (ATS) in iOS 9+
    var allow_navigations = config.getAllowNavigations();

    if (!ats) {
        ats = {};
    }
    
    allow_navigations.forEach(function(allow_navigation) {
        ats = processUrlAsATS(ats, allow_navigation.href, allow_navigation.minimum_tls_version, allow_navigation.requires_forward_secrecy);
    });
    
    return ats;
}

function processUrlAsATS(ats0, url, minimum_tls_version, requires_forward_secrecy) {
    
    var ats = JSON.parse(JSON.stringify(ats0)); // (shallow) copy, to prevent side effects, +testable
    
    if (url === '*') {
        ats['NSAllowsArbitraryLoads'] = true;
        return ats;
    }

    if (!ats['NSExceptionDomains']) {
        ats['NSExceptionDomains'] = {};
    }

    var href = URL.parse(url);
    var includesSubdomains = false;
    var hostname = href.hostname;

    if (!hostname) {
        // check origin, if it allows subdomains (wildcard in hostname), we set NSIncludesSubdomains to YES. Default is NO
        var subdomain1 = '/*.'; // wildcard in hostname
        var subdomain2 = '*://*.'; // wildcard in hostname and protocol
        var subdomain3 = '*://'; // wildcard in protocol only
        if (href.pathname.indexOf(subdomain1) === 0) {
            includesSubdomains = true;
            hostname = href.pathname.substring(subdomain1.length);
        } else if (href.pathname.indexOf(subdomain2) === 0) {
            includesSubdomains = true;
            hostname = href.pathname.substring(subdomain2.length);
        } else if (href.pathname.indexOf(subdomain3) === 0) {
            includesSubdomains = false;
            hostname = href.pathname.substring(subdomain3.length);
        } else {
            // Handling "scheme:*" case to avoid creating of a blank key in NSExceptionDomains.
            return ats;
        }
    }

    // get existing entry, if any
    var exceptionDomain = ats['NSExceptionDomains'][hostname] || {};

    if (includesSubdomains) {
        exceptionDomain['NSIncludesSubdomains'] = true;
    } else {
        delete exceptionDomain['NSIncludesSubdomains'];
    }

    if (minimum_tls_version && minimum_tls_version !== 'TLSv1.2') { // default is TLSv1.2
        exceptionDomain['NSExceptionMinimumTLSVersion'] = minimum_tls_version;
    } else {
        delete exceptionDomain['NSExceptionMinimumTLSVersion'];
    }

    var rfs = (requires_forward_secrecy === 'true');
    if (requires_forward_secrecy && !rfs) { // default is true
        exceptionDomain['NSExceptionRequiresForwardSecrecy'] = rfs;
    } else {
        delete exceptionDomain['NSExceptionRequiresForwardSecrecy'];
    }

    // if the scheme is HTTP, we set NSExceptionAllowsInsecureHTTPLoads to YES. Default is NO
    if (href.protocol === 'http:') {
        exceptionDomain['NSExceptionAllowsInsecureHTTPLoads'] = true;
    }
    else if (!href.protocol && href.pathname.indexOf('*:/') === 0) { // wilcard in protocol
        exceptionDomain['NSExceptionAllowsInsecureHTTPLoads'] = true;
    } else {
        delete exceptionDomain['NSExceptionAllowsInsecureHTTPLoads'];
    }

    ats['NSExceptionDomains'][hostname] = exceptionDomain;

    return ats;
}

function folderExists(folderPath) {
    try {
        var stat = fs.statSync(folderPath);
        return stat && stat.isDirectory();
    } catch (e) {
        return false;
    }
}

// Construct a default value for CFBundleVersion as the version with any
// -rclabel stripped=.
function default_CFBundleVersion(version) {
    return version.split('-')[0];
}

// Converts cordova specific representation of target device to XCode value
function parseTargetDevicePreference(value) {
    if (!value) return null;
    var map = { 'universal': '"1,2"', 'handset': '"1"', 'tablet': '"2"'};
    if (map[value.toLowerCase()]) {
        return map[value.toLowerCase()];
    }
    events.emit('warn', 'Unknown target-device preference value: "' + value + '".');
    return null;
}
