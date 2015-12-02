/*
 *
 * Copyright 2013 Canonical Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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

/* jshint sub:true */

var fs            = require('fs'),
    path          = require('path'),
    util          = require('../util'),
    shell         = require('shelljs'),
    Q             = require('q'),
    Parser        = require('./parser'),
    os            = require('os'),
    ConfigParser = require('cordova-common').ConfigParser;

function ubuntu_parser(project) {

    // Call the base class constructor
    Parser.call(this, 'ubuntu', project);

    this.path = project;
    this.config = new ConfigParser(this.config_xml());
    this.update_manifest();
}

function sanitize(str) {
    return str.replace(/\n/g, ' ').replace(/^\s+|\s+$/g, '');
}

require('util').inherits(ubuntu_parser, Parser);

module.exports = ubuntu_parser;

ubuntu_parser.prototype.update_from_config = function(config) {
    if (config instanceof ConfigParser) {
    } else {
        return Q.reject(new Error('update_from_config requires a ConfigParser object'));
    }

    this.config = new ConfigParser(this.config_xml());
    this.config.setName(config.name());
    this.config.setVersion(config.version());
    this.config.setPackageName(config.packageName());
    this.config.setDescription(config.description());

    this.config.write();

    return this.update_manifest();
};

ubuntu_parser.prototype.cordovajs_path = function(libDir) {
    var jsPath = path.join(libDir, 'www', 'cordova.js');
    return path.resolve(jsPath);
};

ubuntu_parser.prototype.cordovajs_src_path = function(libDir) {
    var jsPath = path.join(libDir, 'cordova-js-src');
    return path.resolve(jsPath);
};

ubuntu_parser.prototype.update_manifest = function() {
    var nodearch2debarch = { 'arm': 'armhf',
                             'ia32': 'i386',
                             'x64': 'amd64'};
    var arch;
    if (os.arch() in nodearch2debarch)
        arch = nodearch2debarch[os.arch()];
    else
        return Q.reject(new Error('unknown cpu arch'));

    if (!this.config.author())
        return Q.reject(new Error('config.xml should contain author'));

    var manifest = { name: this.config.packageName(),
                     version: this.config.version(),
                     title: this.config.name(),
                     hooks: { cordova: { desktop: 'cordova.desktop',
                                         apparmor: 'apparmor.json' } },
                     framework: 'ubuntu-sdk-13.10',
                     maintainer: sanitize(this.config.author())  + ' <' + this.config.doc.find('author').attrib.email + '>',
                     architecture: arch,
                     description: sanitize(this.config.description()) };

    var name = sanitize(this.config.name()); //FIXME: escaping
    var content = '[Desktop Entry]\nName=' + name + '\nExec=./cordova-ubuntu www/\nTerminal=false\nType=Application\nX-Ubuntu-Touch=true';

    if (this.config.doc.find('icon') && this.config.doc.find('icon').attrib.src) {
        var iconPath = path.join(this.path, 'www', this.config.doc.find('icon').attrib.src);
        if (fs.existsSync(iconPath))
            content += '\nIcon=www/' + this.config.doc.find('icon').attrib.src;
        else
            return Q.reject(new Error('icon does not exist: ' + iconPath));
    } else {
        content += '\nIcon=qmlscene';
        console.warn('missing icon element in config.xml');
    }
    fs.writeFileSync(path.join(this.path, 'manifest.json'), JSON.stringify(manifest));
    fs.writeFileSync(path.join(this.path, 'cordova.desktop'), content);

    var policy = { policy_groups: ['networking', 'audio'], policy_version: 1 };

    this.config.doc.getroot().findall('./feature/param').forEach(function (element) {
        if (element.attrib.policy_group && policy.policy_groups.indexOf(element.attrib.policy_group) === -1)
            policy.policy_groups.push(element.attrib.policy_group);
    });

    fs.writeFileSync(path.join(this.path, 'apparmor.json'), JSON.stringify(policy));

    return Q();
};

ubuntu_parser.prototype.config_xml = function(){
    return path.join(this.path, 'config.xml');
};

ubuntu_parser.prototype.www_dir = function() {
    return path.join(this.path, 'www');
};

ubuntu_parser.prototype.update_www = function() {
    var projectRoot = util.isCordova(this.path);
    var www = util.projectWww(projectRoot);

    shell.rm('-rf', this.www_dir());
    shell.cp('-rf', www, this.path);
};

ubuntu_parser.prototype.update_overrides = function() {
    var projectRoot = util.isCordova(this.path);
    var mergesPath = path.join(util.appDir(projectRoot), 'merges', 'ubuntu');
    if(fs.existsSync(mergesPath)) {
        var overrides = path.join(mergesPath, '*');
        shell.cp('-rf', overrides, this.www_dir());
    }
};

// Returns a promise.
ubuntu_parser.prototype.update_project = function(cfg) {
    var self = this;

    return this.update_from_config(cfg)
    .then(function() {
        self.update_overrides();
        util.deleteSvnFolders(self.www_dir());
    });
};
