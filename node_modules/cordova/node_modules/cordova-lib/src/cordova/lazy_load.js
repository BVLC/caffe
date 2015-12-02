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

// The URL:true below prevents jshint error "Redefinition or 'URL'."
/* globals URL:true */

var path          = require('path'),
    _             = require('underscore'),
    fs            = require('fs'),
    shell         = require('shelljs'),
    platforms     = require('../platforms/platforms'),
    npmconf       = require('npmconf'),
    events        = require('cordova-common').events,
    request       = require('request'),
    config        = require('./config'),
    HooksRunner   = require('../hooks/HooksRunner'),
    zlib          = require('zlib'),
    tar           = require('tar'),
    URL           = require('url'),
    Q             = require('q'),
    npm           = require('npm'),
    npmhelper     = require('../util/npm-helper'),
    unpack        = require('../util/unpack'),
    util          = require('./util'),
    gitclone      = require('../gitclone'),
    stubplatform  = {
        url    : undefined,
        version: undefined,
        altplatform: undefined,
        subdirectory: ''
    };

exports.cordova = cordova;
exports.cordova_git = cordova_git;
exports.git_clone = git_clone_platform;
exports.cordova_npm = cordova_npm;
exports.npm_cache_add = npm_cache_add;
exports.custom = custom;
exports.based_on_config = based_on_config;

function Platform(platformString) {
    var name,
        platform,
        parts,
        version;
    if (platformString.indexOf('@') != -1) {
        parts = platformString.split('@');
        name = parts[0];
        version = parts[1];
    } else {
        name = platformString;
    }
    platform = _.extend({}, platforms[name]);
    this.name = name;
    this.version = version || platform.version;
    this.packageName = 'cordova-' + name;
    this.source = 'source' in platform ? platform.source : 'npm';
}

// Returns a promise for the path to the lazy-loaded directory.
function based_on_config(project_root, platform, opts) {
    var custom_path = config.has_custom_path(project_root, platform);
    if (custom_path === false && platform === 'windows') {
        custom_path = config.has_custom_path(project_root, 'windows8');
    }
    if (custom_path) {
        var dot_file = config.read(project_root),
            mixed_platforms = _.extend({}, platforms);
        mixed_platforms[platform] = _.extend({}, mixed_platforms[platform], dot_file.lib && dot_file.lib[platform] || {});
        return module.exports.custom(mixed_platforms, platform);
    } else {
        return module.exports.cordova(platform, opts);
    }
}

// Returns a promise for the path to the lazy-loaded directory.
function cordova(platform, opts) {
    platform = new Platform(platform);
    var use_git = opts && opts.usegit || platform.source === 'git';
    if ( use_git ) {
        return module.exports.cordova_git(platform);
    } else {
        return module.exports.cordova_npm(platform);
    }
}

function cordova_git(platform) {
    var mixed_platforms = _.extend({}, platforms),
        plat;
    if (!(platform.name in platforms)) {
        return Q.reject(new Error('Cordova library "' + platform.name + '" not recognized.'));
    }
    plat = mixed_platforms[platform.name];
    plat.id = 'cordova';

    // We can't use a version range when getting from git, so if we have a range, find the latest release on npm that matches.
    return util.getLatestMatchingNpmVersion(platform.packageName, platform.version).then(function (version) {
        plat.version = version;
        if (/^...*:/.test(plat.url)) {
            plat.url = plat.url + ';a=snapshot;h=' + version + ';sf=tgz';
        }
        return module.exports.custom(mixed_platforms, platform.name);
    });
}

function cordova_npm(platform) {
    if ( !(platform.name in platforms) ) {
        return Q.reject(new Error('Cordova library "' + platform.name + '" not recognized.'));
    }
    // Check if this version was already downloaded from git, if yes, use that copy.
    // TODO: remove this once we fully switch to npm workflow.
    var platdir = platforms[platform.name].altplatform || platform.name;
    // If platform.version specifies a *range*, we need to determine what version we'll actually get from npm (the
    // latest version that matches the range) to know what local directory to look for.
    return util.getLatestMatchingNpmVersion(platform.packageName, platform.version).then(function (version) {
        var git_dload_dir = path.join(util.libDirectory, platdir, 'cordova', version);
        if (fs.existsSync(git_dload_dir)) {
            var subdir = platforms[platform.name].subdirectory;
            if (subdir) {
                git_dload_dir = path.join(git_dload_dir, subdir);
            }
            events.emit('verbose', 'Platform files for "' + platform.name + '" previously downloaded not from npm. Using that copy.');
            return Q(git_dload_dir);
        }

        // Note that because the version of npm we use internally doesn't support caret versions, in order to allow them
        // from the command line and in config.xml, we use the actual version returned by getLatestMatchingNpmVersion().
        var pkg = platform.packageName + '@' + version;
        return exports.npm_cache_add(pkg);
    });
}

// Equivalent to a command like
// npm cache add cordova-android@3.5.0
// Returns a promise that resolves to directory containing the package.
function npm_cache_add(pkg) {
    var npm_cache_dir = path.join(util.libDirectory, 'npm_cache');
    // 'cache-min' is the time in seconds npm considers the files fresh and
    // does not ask the registry if it got a fresher version.
    var platformNpmConfig = {
        'cache-min': 3600*24,
        cache: npm_cache_dir
    };

    return npmhelper.loadWithSettingsThenRestore(platformNpmConfig, function () {
        return Q.ninvoke(npm.commands, 'cache', ['add', pkg])
        .then(function (info) {
            var pkgDir = path.resolve(npm.cache, info.name, info.version, 'package');
            // Unpack the package that was added to the cache (CB-8154)
            var package_tgz = path.resolve(npm.cache, info.name, info.version, 'package.tgz');
            return unpack.unpackTgz(package_tgz, pkgDir);
        });
    });
}

// Returns a promise for the path to the lazy-loaded directory.
function custom(platforms, platform) {
    var plat;
    var id;
    var uri;
    var url;
    var version;
    var subdir;
    var platdir;
    var download_dir;
    var tmp_dir;
    var lib_dir;
    var isUri;
    if (!(platform in platforms)) {
        return Q.reject(new Error('Cordova library "' + platform + '" not recognized.'));
    }

    plat = _.extend({}, stubplatform, platforms[platform]);
    version = plat.version;
    // Older tools can still provide uri (as opposed to url) as part of extra
    // config to create, it should override the default url provided in
    // platfroms.js
    url = plat.uri || plat.url;
    id = plat.id;
    subdir = plat.subdirectory;
    platdir = plat.altplatform || platform;
    // Return early for already-cached remote URL, or for local URLs.
    uri = URL.parse(url);
    isUri = uri.protocol && uri.protocol[1] != ':'; // second part of conditional is for awesome windows support. fuuu windows
    if (isUri) {
        download_dir = path.join(util.libDirectory, platdir, id, version);
        lib_dir = path.join(download_dir, subdir);
        if (fs.existsSync(download_dir)) {
            events.emit('verbose', id + ' library for "' + platform + '" already exists. No need to download. Continuing.');
            return Q(lib_dir);
        }
    } else {
        // Local path.
        lib_dir = path.join(url, subdir);
        return Q(lib_dir);
    }

    return HooksRunner.fire('before_library_download', {
        platform:platform,
        url:url,
        id:id,
        version:version
    }).then(function() {
        var uri = URL.parse(url);
        var d = Q.defer();
        npmconf.load(function(err, conf) {
            // Check if NPM proxy settings are set. If so, include them in the request() call.
            var proxy;
            if (uri.protocol == 'https:') {
                proxy = conf.get('https-proxy');
            } else if (uri.protocol == 'http:') {
                proxy = conf.get('proxy');
            }
            var strictSSL = conf.get('strict-ssl');

            // Create a tmp dir. Using /tmp is a problem because it's often on a different partition and sehll.mv()
            // fails in this case with "EXDEV, cross-device link not permitted".
            var tmp_subidr = 'tmp_' + id + '_' + process.pid + '_' + (new Date()).valueOf();
            tmp_dir = path.join(util.libDirectory, 'tmp', tmp_subidr);
            shell.rm('-rf', tmp_dir);
            shell.mkdir('-p', tmp_dir);

            var size = 0;
            var request_options = {url:url};
            if (proxy) {
                request_options.proxy = proxy;
            }
            if (typeof strictSSL == 'boolean') {
                request_options.strictSSL = strictSSL;
            }
            events.emit('verbose', 'Requesting ' + JSON.stringify(request_options) + '...');
            events.emit('log', 'Downloading ' + id + ' library for ' + platform + '...');
            var req = request.get(request_options, function(err, res, body) {
                if (err) {
                    shell.rm('-rf', tmp_dir);
                    d.reject(err);
                } else if (res.statusCode != 200) {
                    shell.rm('-rf', tmp_dir);
                    d.reject(new Error('HTTP error ' + res.statusCode + ' retrieving version ' + version + ' of ' + id + ' for ' + platform));
                } else {
                    size = body.length;
                }
            });
            req.pipe(zlib.createUnzip())
            .on('error', function(err) {
                // Sometimes if the URL is bad (most likely unavailable version), and git-wip-us.apache.org is
                // particularly slow at responding, we hit an error because of bad data piped to zlib.createUnzip()
                // before we hit the request.get() callback above (with a 404 error). Handle that gracefully. It is
                // likely that we will end up calling d.reject() for an HTTP error in the request() callback above, but
                // in case not, reject with a useful error here.
                d.reject(new Error('Unable to fetch platform ' + platform + '@' + version + ': Error: version not found.'));
            })
            .pipe(tar.Extract({path:tmp_dir}))
            .on('error', function(err) {
                shell.rm('-rf', tmp_dir);
                d.reject(err);
            })
            .on('end', function() {
                events.emit('verbose', 'Downloaded, unzipped and extracted ' + size + ' byte response.');
                events.emit('log', 'Download complete');
                var entries = fs.readdirSync(tmp_dir);
                var entry = path.join(tmp_dir, entries[0]);
                shell.mkdir('-p', download_dir);
                shell.mv('-f', path.join(entry, '*'), download_dir);
                shell.rm('-rf', tmp_dir);
                d.resolve(HooksRunner.fire('after_library_download', {
                    platform:platform,
                    url:url,
                    id:id,
                    version:version,
                    path: lib_dir,
                    size:size,
                    symlink:false
                }));
            });
        });
        return d.promise.then(function () { return lib_dir; });
    });
}

// Returns a promise
function git_clone_platform(git_url, branch) {
    // Create a tmp dir. Using /tmp is a problem because it's often on a different partition and sehll.mv()
    // fails in this case with "EXDEV, cross-device link not permitted".
    var tmp_subidr = 'tmp_cordova_git_' + process.pid + '_' + (new Date()).valueOf();
    var tmp_dir = path.join(util.libDirectory, 'tmp', tmp_subidr);
    shell.rm('-rf', tmp_dir);
    shell.mkdir('-p', tmp_dir);

    return HooksRunner.fire('before_platform_clone', {
        repository: git_url,
        location: tmp_dir
    }).then(function () {
        var branchToCheckout = branch || 'master';
        return gitclone.clone(git_url, branchToCheckout, tmp_dir);
    }).then(function () {
        HooksRunner.fire('after_platform_clone', {
            repository: git_url,
            location: tmp_dir
        });
        return tmp_dir;
    }).fail(function (err) {
        shell.rm('-rf', tmp_dir);
        return Q.reject(err);
    });
}


