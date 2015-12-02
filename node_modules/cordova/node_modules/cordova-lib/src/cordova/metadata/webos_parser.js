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

/* jshint node:true, bitwise:true, undef:true, trailing:true, quotmark:true,
          indent:4, unused:vars, latedef:nofunc,
          sub:true
*/

var fs = require('fs'),
    path = require('path'),
    shell = require('shelljs'),
    util = require('../util'),
    Q = require('q'),
    Parser = require('./parser');

function webos_parser(project) {
    // Call the base class constructor
    Parser.call(this, 'webos', project);
    this.path = project;
}

require('util').inherits(webos_parser, Parser);

module.exports = webos_parser;

// Returns a promise.
webos_parser.prototype.update_from_config = function(config) {
    var www = this.www_dir();
    var manifestPath = path.join(www, 'appinfo.json');
    var manifest = {type: 'web', uiRevision:2};

    // Load existing manifest
    if (fs.existsSync(manifestPath)) {
        manifest = JSON.parse(fs.readFileSync(manifestPath));
    }

    // overwrite properties existing in config.xml
    manifest.id = config.packageName() || 'org.apache.cordova.example';
    var contentNode = config.doc.find('content');
    var contentSrc = contentNode && contentNode.attrib['src'] || 'index.html';
    manifest.main = contentSrc;
    manifest.version = config.version() || '0.0.1';
    manifest.title = config.name() || 'CordovaExample';
    manifest.appDescription = config.description() || '';
    manifest.vendor = config.author() || 'My Company';

    var authorNode = config.doc.find('author');
    var authorUrl = authorNode && authorNode.attrib['href'];
    if (authorUrl) {
        manifest.vendorurl = authorUrl;
    }

    var projectRoot = util.isCordova(this.path);
    var copyImg = function(src, type) {
        var index = src.indexOf('www');
        if(index===0 || index===1) {
            return src.substring(index+4);
        } else {
            var newSrc = type + '.png';
            shell.cp('-f', path.join(projectRoot, src), path.join(www, newSrc));
            return newSrc;
        }
    };

    var icons = config.getIcons('webos');
    // if there are icon elements in config.xml
    if (icons) {
        var setIcon = function(type, size) {
            var item = icons.getBySize(size, size);
            if(item && item.src) {
                manifest[type] = copyImg(item.src, type);
            } else {
                item = icons.getDefault();
                if(item && item.src) {
                    manifest[type] = copyImg(item.src, type);
                }
            }
        };
        setIcon('icon', 80, 80);
        setIcon('largeIcon', 130, 130);
    }

    var splash = config.getSplashScreens('webos');
    // if there are icon elements in config.xml
    if (splash) {
        var splashImg = splash.getBySize(1920, 1080);
        if(splashImg && splashImg.src) {
            manifest.splashBackground = copyImg(splashImg.src, 'splash');
        }
    }

    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, '\t'));

    return Q();
};

webos_parser.prototype.www_dir = function() {
    return path.join(this.path, 'www');
};

// Used for creating platform_www in projects created by older versions.
webos_parser.prototype.cordovajs_path = function(libDir) {
    var jsPath = path.join(libDir, 'cordova-lib', 'cordova.js');
    return path.resolve(jsPath);
};

webos_parser.prototype.cordovajs_src_path = function(libDir) {
    var jsPath = path.join(libDir, 'cordova-js-src');
    return path.resolve(jsPath);
};

// Replace the www dir with contents of platform_www and app www.
webos_parser.prototype.update_www = function() {
    var projectRoot = util.isCordova(this.path);
    var app_www = util.projectWww(projectRoot);
    var platform_www = path.join(this.path, 'platform_www');

    // Clear the www dir
    shell.rm('-rf', this.www_dir());
    shell.mkdir(this.www_dir());
    // Copy over all app www assets
    if(fs.lstatSync(app_www).isSymbolicLink()) {
        var real_www = fs.realpathSync(app_www);
        if(fs.existsSync(path.join(real_www, 'build/enyo.js'))) {
            // symlinked Enyo bootplate; resolve to bootplate root for
            // ares-webos-sdk to handle the minification
            if(fs.existsSync(path.join(real_www, '../enyo'))) {
                app_www = path.join(real_www, '..');
            } else if (fs.existsSync(path.join(real_www, '../../enyo'))) {
                app_www = path.join(real_www, '../..');
            }
            //double check existence of deploy
            if(!fs.existsSync(path.join(app_www, 'deploy'))) {
                app_www = real_www; //fallback
            }
        }
    }
    shell.cp('-rf', path.join(app_www, '*'), this.www_dir());
    // Copy over stock platform www assets (cordova.js)
    shell.cp('-rf', path.join(platform_www, '*'), this.www_dir());

    // prepare and update deploy.json for cordova components
    var deploy = path.join(this.www_dir(), 'deploy.json');
    if(fs.existsSync(deploy)) {
        try {
            // make stub file entries to guarantee the dir/files are there
            shell.mkdir('-p', path.join(this.www_dir(), 'plugins'));
            var pluginFile = path.join(this.www_dir(), 'cordova_plugins.js');
            if(!fs.existsSync(pluginFile)) {
                fs.writeFileSync(pluginFile, '');
            }
            // add to json if not already there, so they don't get minified out during build
            var obj = JSON.parse(fs.readFileSync(deploy, {encoding:'utf8'}));
            obj.assets = obj.assets || [];
            var assets = ['plugins', 'cordova.js', 'cordova_plugins.js'];
            for(var i=0; i<assets.length; i++) {
                var index = obj.assets.indexOf(assets[i]);
                if(index<0) {
                    index = obj.assets.indexOf('./' + assets[i]);
                }
                if(index<0) {
                    obj.assets.push('./' + assets[i]);
                }
                fs.writeFileSync(deploy, JSON.stringify(obj, null, '\t'));
            }
        } catch(e) {
            console.error('Unable to update deploy.json: ' + e);
        }
    }
};

webos_parser.prototype.update_overrides = function() {
    var projectRoot = util.isCordova(this.path);
    var mergesPath = path.join(util.appDir(projectRoot), 'merges', 'webosos');
    if(fs.existsSync(mergesPath)) {
        var overrides = path.join(mergesPath, '*');
        shell.cp('-rf', overrides, this.www_dir());
    }
};

webos_parser.prototype.config_xml = function(){
    return path.join(this.path, 'config.xml');
};

// Returns a promise.
webos_parser.prototype.update_project = function(cfg) {
    return this.update_from_config(cfg)
        .then(function(){
            this.update_overrides();
            util.deleteSvnFolders(this.www_dir());
        }.bind(this));
};


