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

var path          = require('path'),
    fs            = require('fs'),
    shell         = require('shelljs'),
    events        = require('cordova-common').events,
    config        = require('./config'),
    lazy_load     = require('./lazy_load'),
    Q             = require('q'),
    CordovaError  = require('cordova-common').CordovaError,
    ConfigParser  = require('cordova-common').ConfigParser,
    cordova_util  = require('./util'),
    validateIdentifier = require('valid-identifier');

/**
 * Usage:
 * @dir - directory where the project will be created. Required.
 * @optionalId - app id. Optional.
 * @optionalName - app name. Optional.
 * @cfg - extra config to be saved in .cordova/config.json
 **/
// Returns a promise.
module.exports = create;
function create(dir, optionalId, optionalName, cfg) {
    var argumentCount = arguments.length;
    
    return Q.fcall(function() {
        // Lets check prerequisites first

        if (argumentCount == 3) {
          cfg = optionalName;
          optionalName = undefined;
        } else if (argumentCount == 2) {
          cfg = optionalId;
          optionalId = undefined;
          optionalName = undefined;
        }

        if (!dir) {
            throw new CordovaError('At least the dir must be provided to create new project. See `' + cordova_util.binname + ' help`.');
        }

        //read projects .cordova/config.json file for project settings
        var configFile = config.read(dir);

        //if data exists in the configFile, lets combine it with cfg
        //cfg values take priority over config file
        if(configFile) {
            var finalConfig = {};
            for(var key1 in configFile) {
                finalConfig[key1] = configFile[key1];
            }

            for(var key2 in cfg) {
                finalConfig[key2] = cfg[key2];
            }

            cfg = finalConfig;
        }

        if (!cfg) {
            throw new CordovaError('Must provide a project configuration.');
        } else if (typeof cfg == 'string') {
            cfg = JSON.parse(cfg);
        }

        if (optionalId) cfg.id = optionalId;
        if (optionalName) cfg.name = optionalName;

        // Make absolute.
        dir = path.resolve(dir);

        // dir must be either empty except for .cordova config file or not exist at all..
        var sanedircontents = function (d) {
            var contents = fs.readdirSync(d);
            if (contents.length === 0) {
                return true;
            } else if (contents.length == 1) {
                if (contents[0] == '.cordova') {
                    return true;
                }
            }
            return false;
        };

        if (fs.existsSync(dir) && !sanedircontents(dir)) {
            throw new CordovaError('Path already exists and is not empty: ' + dir);
        }

        if (cfg.id && !validateIdentifier(cfg.id)) {
            throw new CordovaError('App id contains a reserved word, or is not a valid identifier.');
        }


        // This was changed from "uri" to "url", but checking uri for backwards compatibility.
        cfg.lib = cfg.lib || {};
        cfg.lib.www = cfg.lib.www || {};
        cfg.lib.www.url = cfg.lib.www.url || cfg.lib.www.uri;

        if (!cfg.lib.www.url) {
            try {
                cfg.lib.www.url = require('cordova-app-hello-world').dirname;
            } catch (e) {
                // Falling back on npm@2 path hierarchy
                // TODO: Remove fallback after cordova-app-hello-world release
                cfg.lib.www.url = path.join(__dirname, '..', '..', 'node_modules', 'cordova-app-hello-world');
            }
        }

        // TODO (kamrik): extend lazy_load for retrieval without caching to allow net urls for --src.
        cfg.lib.www.version = cfg.lib.www.version || 'not_versioned';
        cfg.lib.www.id = cfg.lib.www.id || 'dummy_id';

        // Make sure that the source www/ is not a direct ancestor of the
        // target www/, or else we will recursively copy forever. To do this,
        // we make sure that the shortest relative path from source-to-target
        // must start by going up at least one directory or with a drive
        // letter for Windows.
        var rel_path = path.relative(cfg.lib.www.url, dir);
        var goes_up = rel_path.split(path.sep)[0] == '..';

        if (!(goes_up || rel_path[1] == ':')) {
            throw new CordovaError(
                'Project dir "' + dir +
                '" must not be created at/inside the template used to create the project "' +
                cfg.lib.www.url + '".'
            );
        }
    })
    .then(function() {
        // Finally, Ready to start!
        events.emit('log', 'Creating a new cordova project.');
    })
    .then(function() {
        // Strip link and url from cfg to avoid them being persisted to disk via .cordova/config.json.
        // TODO: apparently underscore has no deep clone.  Replace with lodash or something. For now, abuse JSON.
        var cfgToPersistToDisk = JSON.parse(JSON.stringify(cfg));

        delete cfgToPersistToDisk.lib.www;
        if (Object.keys(cfgToPersistToDisk.lib).length === 0) {
            delete cfgToPersistToDisk.lib;
        }

        // Update cached version of config.json
        var origAutoPersist = config.getAutoPersist();
        config.setAutoPersist(false);
        config(dir, cfgToPersistToDisk);
        config.setAutoPersist(origAutoPersist);
    })
    .then(function() {
        if (!!cfg.lib.www.link) {
            events.emit('verbose', 'Symlinking assets."');
            return cfg.lib.www.url;
        } else {
            events.emit('verbose', 'Copying assets."');
            return lazy_load.custom({ 'www': cfg.lib.www }, 'www')
            .fail(function (error) {
                var message = 'Failed to fetch custom www assets from ' + cfg.lib.www +
                    '\nProbably this is either a connection problem, or assets URL is incorrect.' +
                    '\nCheck your connection and assets URL.' +
                    '\n' + error;
                return Q.reject(message);
            });
        }
    })
    .then(function(import_from_path) {
        if (!fs.existsSync(import_from_path)) {
            throw new CordovaError('Could not find directory: ' + import_from_path);
        }

        var paths = {
            root: import_from_path,
            www: import_from_path
        };

        // Keep going into child "www" folder if exists in stock app package.
        while (fs.existsSync(path.join(paths.www, 'www'))) {
            paths.root = paths.www;
            paths.www = path.join(paths.root, 'www');
        }

        if (fs.existsSync(path.join(paths.root, 'config.xml'))) {
            paths.configXml = path.join(paths.root, 'config.xml');
            paths.configXmlLinkable = true;
        } else {
            try {
                paths.configXml = path.join(require('cordova-app-hello-world').dirname, 'config.xml');
            } catch (e) {
                // Falling back on npm@2 path hierarchy
                // TODO: Remove fallback after cordova-app-hello-world release
                paths.configXml = path.join(__dirname, '..', '..', 'node_modules', 'cordova-app-hello-world', 'config.xml');
            }
        }
        if (fs.existsSync(path.join(paths.root, 'merges'))) {
            paths.merges = path.join(paths.root, 'merges');
        } else {
            // No merges by default
        }
        if (fs.existsSync(path.join(paths.root, 'hooks'))) {
            paths.hooks = path.join(paths.root, 'hooks');
            paths.hooksLinkable = true;
        } else {
            try {
                paths.hooks = path.join(require('cordova-app-hello-world').dirname, 'hooks');
            } catch (e) {
                // Falling back on npm@2 path hierarchy
                // TODO: Remove fallback after cordova-app-hello-world release
                paths.hooks = path.join(__dirname, '..', '..', 'node_modules', 'cordova-app-hello-world', 'hooks');
            }
        }

        var dirAlreadyExisted = fs.existsSync(dir);
        if (!dirAlreadyExisted) {
            fs.mkdirSync(dir);
        }

        var tryToLink = !!cfg.lib.www.link;
        function copyOrLink(src, dst, linkable) {
            if (src) {
                if (tryToLink && linkable) {
                    fs.symlinkSync(src, dst, 'dir');
                } else {
                    shell.mkdir(dst);
                    shell.cp('-R', path.join(src, '*'), dst);
                }
            }
        }
        try {
            copyOrLink(paths.www, path.join(dir, 'www'), true);
            copyOrLink(paths.merges, path.join(dir, 'merges'), true);
            copyOrLink(paths.hooks, path.join(dir, 'hooks'), paths.hooksLinkable);
            if (paths.configXml) {
                if (tryToLink && paths.configXmlLinkable) {
                    fs.symlinkSync(paths.configXml, path.join(dir, 'config.xml'));
                } else {
                    shell.cp(paths.configXml, path.join(dir, 'config.xml'));
                }
            }
        } catch (e) {
            if (!dirAlreadyExisted) {
                shell.rm('-rf', dir);
            }
            if (process.platform.slice(0, 3) == 'win' && e.code == 'EPERM')  {
                throw new CordovaError('Symlinks on Windows require Administrator privileges');
            }
            throw e;
        }

        // Create basic project structure.
        shell.mkdir(path.join(dir, 'platforms'));
        shell.mkdir(path.join(dir, 'plugins'));

        // Write out id and name to config.xml
        var configPath = cordova_util.projectConfig(dir);
        var conf = new ConfigParser(configPath);
        if (cfg.id) conf.setPackageName(cfg.id);
        if (cfg.name) conf.setName(cfg.name);
        conf.write();
    });
}
