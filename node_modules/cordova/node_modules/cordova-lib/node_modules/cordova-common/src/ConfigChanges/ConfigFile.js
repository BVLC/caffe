/*
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

var fs = require('fs');
var path = require('path');

var bplist = require('bplist-parser');
var et   = require('elementtree');
var glob = require('glob');
var plist = require('plist');

var plist_helpers = require('../util/plist-helpers');
var xml_helpers = require('../util/xml-helpers');

/******************************************************************************
* ConfigFile class
*
* Can load and keep various types of config files. Provides some functionality
* specific to some file types such as grafting XML children. In most cases it
* should be instantiated by ConfigKeeper.
*
* For plugin.xml files use as:
* plugin_config = self.config_keeper.get(plugin_dir, '', 'plugin.xml');
*
* TODO: Consider moving it out to a separate file and maybe partially with
* overrides in platform handlers.
******************************************************************************/
function ConfigFile(project_dir, platform, file_tag) {
    this.project_dir = project_dir;
    this.platform = platform;
    this.file_tag = file_tag;
    this.is_changed = false;

    this.load();
}

// ConfigFile.load()
ConfigFile.prototype.load = ConfigFile_load;
function ConfigFile_load() {
    var self = this;

    // config file may be in a place not exactly specified in the target
    var filepath = self.filepath = resolveConfigFilePath(self.project_dir, self.platform, self.file_tag);

    if ( !filepath || !fs.existsSync(filepath) ) {
        self.exists = false;
        return;
    }
    self.exists = true;
    self.mtime = fs.statSync(self.filepath).mtime;

    var ext = path.extname(filepath);
    // Windows8 uses an appxmanifest, and wp8 will likely use
    // the same in a future release
    if (ext == '.xml' || ext == '.appxmanifest') {
        self.type = 'xml';
        self.data = xml_helpers.parseElementtreeSync(filepath);
    } else {
        // plist file
        self.type = 'plist';
        // TODO: isBinaryPlist() reads the file and then parse re-reads it again.
        //       We always write out text plist, not binary.
        //       Do we still need to support binary plist?
        //       If yes, use plist.parseStringSync() and read the file once.
        self.data = isBinaryPlist(filepath) ?
                bplist.parseBuffer(fs.readFileSync(filepath)) :
                plist.parse(fs.readFileSync(filepath, 'utf8'));
    }
}

ConfigFile.prototype.save = function ConfigFile_save() {
    var self = this;
    if (self.type === 'xml') {
        fs.writeFileSync(self.filepath, self.data.write({indent: 4}), 'utf-8');
    } else {
        // plist
        var regExp = new RegExp('<string>[ \t\r\n]+?</string>', 'g');
        fs.writeFileSync(self.filepath, plist.build(self.data).replace(regExp, '<string></string>'));
    }
    self.is_changed = false;
};

ConfigFile.prototype.graft_child = function ConfigFile_graft_child(selector, xml_child) {
    var self = this;
    var filepath = self.filepath;
    var result;
    if (self.type === 'xml') {
        var xml_to_graft = [et.XML(xml_child.xml)];
        result = xml_helpers.graftXML(self.data, xml_to_graft, selector, xml_child.after);
        if ( !result) {
            throw new Error('grafting xml at selector "' + selector + '" from "' + filepath + '" during config install went bad :(');
        }
    } else {
        // plist file
        result = plist_helpers.graftPLIST(self.data, xml_child.xml, selector);
        if ( !result ) {
            throw new Error('grafting to plist "' + filepath + '" during config install went bad :(');
        }
    }
    self.is_changed = true;
};

ConfigFile.prototype.prune_child = function ConfigFile_prune_child(selector, xml_child) {
    var self = this;
    var filepath = self.filepath;
    var result;
    if (self.type === 'xml') {
        var xml_to_graft = [et.XML(xml_child.xml)];
        result = xml_helpers.pruneXML(self.data, xml_to_graft, selector);
    } else {
        // plist file
        result = plist_helpers.prunePLIST(self.data, xml_child.xml, selector);
    }
    if (!result) {
        var err_msg = 'Pruning at selector "' + selector + '" from "' + filepath + '" went bad.';
        throw new Error(err_msg);
    }
    self.is_changed = true;
};

// Some config-file target attributes are not qualified with a full leading directory, or contain wildcards.
// Resolve to a real path in this function.
// TODO: getIOSProjectname is slow because of glob, try to avoid calling it several times per project.
function resolveConfigFilePath(project_dir, platform, file) {
    var filepath = path.join(project_dir, file);
    var matches;

    if (file.indexOf('*') > -1) {
        // handle wildcards in targets using glob.
        matches = glob.sync(path.join(project_dir, '**', file));
        if (matches.length) filepath = matches[0];

        // [CB-5989] multiple Info.plist files may exist. default to $PROJECT_NAME-Info.plist
        if(matches.length > 1 && file.indexOf('-Info.plist')>-1){
            var plistName =  getIOSProjectname(project_dir)+'-Info.plist';
            for (var i=0; i < matches.length; i++) {
                if(matches[i].indexOf(plistName) > -1){
                    filepath = matches[i];
                    break;
                }
            }
        }
        return filepath;
    }

    // special-case config.xml target that is just "config.xml". This should be resolved to the real location of the file.
    // TODO: move the logic that contains the locations of config.xml from cordova CLI into plugman.
    if (file == 'config.xml') {
        if (platform == 'ubuntu') {
            filepath = path.join(project_dir, 'config.xml');
        } else if (platform == 'ios') {
            var iospath = getIOSProjectname(project_dir);
            filepath = path.join(project_dir,iospath, 'config.xml');
        } else if (platform == 'android') {
            filepath = path.join(project_dir, 'res', 'xml', 'config.xml');
        } else {
            matches = glob.sync(path.join(project_dir, '**', 'config.xml'));
            if (matches.length) filepath = matches[0];
        }
        return filepath;
    }

    // None of the special cases matched, returning project_dir/file.
    return filepath;
}

// Find out the real name of an iOS project
// TODO: glob is slow, need a better way or caching, or avoid using more than once.
function getIOSProjectname(project_dir) {
    var matches = glob.sync(path.join(project_dir, '*.xcodeproj'));
    var iospath;
    if (matches.length === 1) {
        iospath = path.basename(matches[0],'.xcodeproj');
    } else {
        var msg;
        if (matches.length === 0) {
            msg = 'Does not appear to be an xcode project, no xcode project file in ' + project_dir;
        } else {
            msg = 'There are multiple *.xcodeproj dirs in ' + project_dir;
        }
        throw new Error(msg);
    }
    return iospath;
}

// determine if a plist file is binary
function isBinaryPlist(filename) {
    // I wish there was a synchronous way to read only the first 6 bytes of a
    // file. This is wasteful :/
    var buf = '' + fs.readFileSync(filename, 'utf8');
    // binary plists start with a magic header, "bplist"
    return buf.substring(0, 6) === 'bplist';
}

module.exports = ConfigFile;
