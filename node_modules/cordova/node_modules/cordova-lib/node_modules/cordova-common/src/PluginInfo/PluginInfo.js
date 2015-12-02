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

/* jshint sub:true, laxcomma:true, laxbreak:true */

/*
A class for holidng the information currently stored in plugin.xml
It should also be able to answer questions like whether the plugin
is compatible with a given engine version.

TODO (kamrik): refactor this to not use sync functions and return promises.
*/


var path = require('path')
  , fs = require('fs')
  , xml_helpers = require('../util/xml-helpers')
  , CordovaError = require('../CordovaError/CordovaError')
  ;

function PluginInfo(dirname) {
    var self = this;

    // METHODS
    // Defined inside the constructor to avoid the "this" binding problems.

    // <preference> tag
    // Example: <preference name="API_KEY" />
    // Used to require a variable to be specified via --variable when installing the plugin.
    self.getPreferences = getPreferences;
    function getPreferences(platform) {
        var arprefs = _getTags(self._et, 'preference', platform, _parsePreference);

        var prefs= {};
        for(var i in arprefs)
        {
            var pref=arprefs[i];
            prefs[pref.preference]=pref.default;
        }
        // returns { key : default | null}
        return prefs;
    }

    function _parsePreference(prefTag) {
        var name = prefTag.attrib.name.toUpperCase();
        var def = prefTag.attrib.default || null;
        return {preference: name, default: def};
    }

    // <asset>
    self.getAssets = getAssets;
    function getAssets(platform) {
        var assets = _getTags(self._et, 'asset', platform, _parseAsset);
        return assets;
    }

    function _parseAsset(tag) {
        var src = tag.attrib.src;
        var target = tag.attrib.target;

        if ( !src || !target) {
            var msg =
                'Malformed <asset> tag. Both "src" and "target" attributes'
                + 'must be specified in\n'
                + self.filepath
                ;
            throw new Error(msg);
        }

        var asset = {
            itemType: 'asset',
            src: src,
            target: target
        };
        return asset;
    }


    // <dependency>
    // Example:
    // <dependency id="com.plugin.id"
    //     url="https://github.com/myuser/someplugin"
    //     commit="428931ada3891801"
    //     subdir="some/path/here" />
    self.getDependencies = getDependencies;
    function getDependencies(platform) {
        var deps = _getTags(
                self._et,
                'dependency',
                platform,
                _parseDependency
        );
        return deps;
    }

    function _parseDependency(tag) {
        var dep =
            { id : tag.attrib.id
            , url : tag.attrib.url || ''
            , subdir : tag.attrib.subdir || ''
            , commit : tag.attrib.commit
            };

        dep.git_ref = dep.commit;

        if ( !dep.id ) {
            var msg =
                '<dependency> tag is missing id attribute in '
                + self.filepath
                ;
            throw new CordovaError(msg);
        }
        return dep;
    }


    // <config-file> tag
    self.getConfigFiles = getConfigFiles;
    function getConfigFiles(platform) {
        var configFiles = _getTags(self._et, 'config-file', platform, _parseConfigFile);
        return configFiles;
    }

    function _parseConfigFile(tag) {
        var configFile =
            { target : tag.attrib['target']
            , parent : tag.attrib['parent']
            , after : tag.attrib['after']
            , xmls : tag.getchildren()
            // To support demuxing via versions
            , versions : tag.attrib['versions']
            , deviceTarget: tag.attrib['device-target']
            };
        return configFile;
    }

    // <info> tags, both global and within a <platform>
    // TODO (kamrik): Do we ever use <info> under <platform>? Example wanted.
    self.getInfo = getInfo;
    function getInfo(platform) {
        var infos = _getTags(
                self._et,
                'info',
                platform,
                function(elem) { return elem.text; }
        );
        // Filter out any undefined or empty strings.
        infos = infos.filter(Boolean);
        return infos;
    }

    // <source-file>
    // Examples:
    // <source-file src="src/ios/someLib.a" framework="true" />
    // <source-file src="src/ios/someLib.a" compiler-flags="-fno-objc-arc" />
    self.getSourceFiles = getSourceFiles;
    function getSourceFiles(platform) {
        var sourceFiles = _getTagsInPlatform(self._et, 'source-file', platform, _parseSourceFile);
        return sourceFiles;
    }

    function _parseSourceFile(tag) {
        return {
            itemType: 'source-file',
            src: tag.attrib.src,
            framework: isStrTrue(tag.attrib.framework),
            weak: isStrTrue(tag.attrib.weak),
            compilerFlags: tag.attrib['compiler-flags'],
            targetDir: tag.attrib['target-dir']
        };
    }

    // <header-file>
    // Example:
    // <header-file src="CDVFoo.h" />
    self.getHeaderFiles = getHeaderFiles;
    function getHeaderFiles(platform) {
        var headerFiles = _getTagsInPlatform(self._et, 'header-file', platform, function(tag) {
            return {
                itemType: 'header-file',
                src: tag.attrib.src,
                targetDir: tag.attrib['target-dir']
            };
        });
        return headerFiles;
    }

    // <resource-file>
    // Example:
    // <resource-file src="FooPluginStrings.xml" target="res/values/FooPluginStrings.xml" device-target="win" arch="x86" versions="&gt;=8.1" />
    self.getResourceFiles = getResourceFiles;
    function getResourceFiles(platform) {
        var resourceFiles = _getTagsInPlatform(self._et, 'resource-file', platform, function(tag) {
            return {
                itemType: 'resource-file',
                src: tag.attrib.src,
                target: tag.attrib.target,
                versions: tag.attrib.versions,
                deviceTarget: tag.attrib['device-target'],
                arch: tag.attrib.arch
            };
        });
        return resourceFiles;
    }

    // <lib-file>
    // Example:
    // <lib-file src="src/BlackBerry10/native/device/libfoo.so" arch="device" />
    self.getLibFiles = getLibFiles;
    function getLibFiles(platform) {
        var libFiles = _getTagsInPlatform(self._et, 'lib-file', platform, function(tag) {
            return {
                itemType: 'lib-file',
                src: tag.attrib.src,
                arch: tag.attrib.arch,
                Include: tag.attrib.Include,
                versions: tag.attrib.versions,
                deviceTarget: tag.attrib['device-target'] || tag.attrib.target
            };
        });
        return libFiles;
    }

    // <hook>
    // Example:
    // <hook type="before_build" src="scripts/beforeBuild.js" />
    self.getHookScripts = getHookScripts;
    function getHookScripts(hook, platforms) {
        var scriptElements =  self._et.findall('./hook');

        if(platforms) {
            platforms.forEach(function (platform) {
                scriptElements = scriptElements.concat(self._et.findall('./platform[@name="' + platform + '"]/hook'));
            });
        }

        function filterScriptByHookType(el) {
            return el.attrib.src && el.attrib.type && el.attrib.type.toLowerCase() === hook;
        }

        return scriptElements.filter(filterScriptByHookType);
    }

    self.getJsModules = getJsModules;
    function getJsModules(platform) {
        var modules = _getTags(self._et, 'js-module', platform, _parseJsModule);
        return modules;
    }

    function _parseJsModule(tag) {
        var ret = {
            itemType: 'js-module',
            name: tag.attrib.name,
            src: tag.attrib.src,
            clobbers: tag.findall('clobbers').map(function(tag) { return { target: tag.attrib.target }; }),
            merges: tag.findall('merges').map(function(tag) { return { target: tag.attrib.target }; }),
            runs: tag.findall('runs').length > 0
        };

        return ret;
    }

    self.getEngines = function() {
        return self._et.findall('engines/engine').map(function(n) {
            return {
                name: n.attrib.name,
                version: n.attrib.version,
                platform: n.attrib.platform,
                scriptSrc: n.attrib.scriptSrc
            };
        });
    };

    self.getPlatforms = function() {
        return self._et.findall('platform').map(function(n) {
            return { name: n.attrib.name };
        });
    };

    self.getPlatformsArray = function() {
        return self._et.findall('platform').map(function(n) {
            return n.attrib.name;
        });
    };
    self.getFrameworks = function(platform) {
        return _getTags(self._et, 'framework', platform, function(el) {
            var ret = {
                itemType: 'framework',
                type: el.attrib.type,
                parent: el.attrib.parent,
                custom: isStrTrue(el.attrib.custom),
                src: el.attrib.src,
                weak: isStrTrue(el.attrib.weak),
                versions: el.attrib.versions,
                targetDir: el.attrib['target-dir'],
                deviceTarget: el.attrib['device-target'] || el.attrib.target,
                arch: el.attrib.arch
            };
            return ret;
        });
    };

    self.getFilesAndFrameworks = getFilesAndFrameworks;
    function getFilesAndFrameworks(platform) {
        // Please avoid changing the order of the calls below, files will be
        // installed in this order.
        var items = [].concat(
            self.getSourceFiles(platform),
            self.getHeaderFiles(platform),
            self.getResourceFiles(platform),
            self.getFrameworks(platform),
            self.getLibFiles(platform)
        );
        return items;
    }
    ///// End of PluginInfo methods /////


    ///// PluginInfo Constructor logic  /////
    self.filepath = path.join(dirname, 'plugin.xml');
    if (!fs.existsSync(self.filepath)) {
        throw new CordovaError('Cannot find plugin.xml for plugin \'' + path.basename(dirname) + '\'. Please try adding it again.');
    }

    self.dir = dirname;
    var et = self._et = xml_helpers.parseElementtreeSync(self.filepath);
    var pelem = et.getroot();
    self.id = pelem.attrib.id;
    self.version = pelem.attrib.version;

    // Optional fields
    self.name = pelem.findtext('name');
    self.description = pelem.findtext('description');
    self.license = pelem.findtext('license');
    self.repo = pelem.findtext('repo');
    self.issue = pelem.findtext('issue');
    self.keywords = pelem.findtext('keywords');
    self.info = pelem.findtext('info');
    if (self.keywords) {
        self.keywords = self.keywords.split(',').map( function(s) { return s.trim(); } );
    }
    self.getKeywordsAndPlatforms = function () {
        var ret = self.keywords || [];
        return ret.concat('ecosystem:cordova').concat(addCordova(self.getPlatformsArray()));
    };
}  // End of PluginInfo constructor.

// Helper function used to prefix every element of an array with cordova-
// Useful when we want to modify platforms to be cordova-platform
function addCordova(someArray) {
    var newArray = someArray.map(function(element) {
        return 'cordova-' + element;
    });
    return newArray;
}

// Helper function used by most of the getSomething methods of PluginInfo.
// Get all elements of a given name. Both in root and in platform sections
// for the given platform. If transform is given and is a function, it is
// applied to each element.
function _getTags(pelem, tag, platform, transform) {
    var platformTag = pelem.find('./platform[@name="' + platform + '"]');
    if (platform == 'windows' && !platformTag) {
        platformTag = pelem.find('platform[@name="' + 'windows8' + '"]');
    }
    var tagsInRoot = pelem.findall(tag);
    tagsInRoot = tagsInRoot || [];
    var tagsInPlatform = platformTag ? platformTag.findall(tag) : [];
    var tags = tagsInRoot.concat(tagsInPlatform);
    if ( typeof transform === 'function' ) {
        tags = tags.map(transform);
    }
    return tags;
}

// Same as _getTags() but only looks inside a platfrom section.
function _getTagsInPlatform(pelem, tag, platform, transform) {
    var platformTag = pelem.find('./platform[@name="' + platform + '"]');
    if (platform == 'windows' && !platformTag) {
        platformTag = pelem.find('platform[@name="' + 'windows8' + '"]');
    }
    var tags = platformTag ? platformTag.findall(tag) : [];
    if ( typeof transform === 'function' ) {
        tags = tags.map(transform);
    }
    return tags;
}

// Check if x is a string 'true'.
function isStrTrue(x) {
    return String(x).toLowerCase() == 'true';
}

module.exports = PluginInfo;
// Backwards compat:
PluginInfo.PluginInfo = PluginInfo;
PluginInfo.loadPluginsDir = function(dir) {
    var PluginInfoProvider = require('./PluginInfoProvider');
    return new PluginInfoProvider().getAllWithinSearchPath(dir);
};
