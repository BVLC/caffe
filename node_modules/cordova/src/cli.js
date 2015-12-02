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
          laxcomma:true
*/


var path = require('path'),
    fs = require('fs'),
    help = require('./help'),
    nopt,
    _,
    updateNotifier,
    pkg = require('../package.json'),
    logger = require('./logger');

var cordova_lib = require('cordova-lib'),
    CordovaError = cordova_lib.CordovaError,
    cordova = cordova_lib.cordova,
    events = cordova_lib.events;


/*
 * init
 *
 * initializes nopt and underscore
 * nopt and underscore are require()d in try-catch below to print a nice error
 * message if one of them is not installed.
 */
function init() {
    try {
        nopt = require('nopt');
        _ = require('underscore');
        updateNotifier = require('update-notifier');
    } catch (e) {
        console.error(
            'Please run npm install from this directory:\n\t' +
            path.dirname(__dirname)
        );
        process.exit(2);
    }
}

function checkForUpdates() {
    // Checks for available update and returns an instance
    var notifier = updateNotifier({
        pkg: pkg
    });

    // Notify using the built-in convenience method
    notifier.notify();
}

module.exports = cli;
function cli(inputArgs) {
    // When changing command line arguments, update doc/help.txt accordingly.
    var knownOpts =
        { 'verbose' : Boolean
        , 'version' : Boolean
        , 'help' : Boolean
        , 'silent' : Boolean
        , 'experimental' : Boolean
        , 'noregistry' : Boolean
        , 'shrinkwrap' : Boolean
        , 'usegit' : Boolean
        , 'copy-from' : String
        , 'link-to' : path
        , 'searchpath' : String
        , 'variable' : Array
        , 'link': Boolean
        // Flags to be passed to `cordova build/run/emulate`
        , 'debug' : Boolean
        , 'release' : Boolean
        , 'archs' : String
        , 'device' : Boolean
        , 'emulator': Boolean
        , 'target' : String
        , 'browserify': Boolean
        , 'nobuild': Boolean
        , 'list': Boolean
        , 'buildConfig' : String
        };

    var shortHands =
        { 'd' : '--verbose'
        , 'v' : '--version'
        , 'h' : '--help'
        , 'src' : '--copy-from'
        };

    // If no inputArgs given, use process.argv.
    inputArgs = inputArgs || process.argv;

    init();

    checkForUpdates();

    var args = nopt(knownOpts, shortHands, inputArgs);


    if (args.version) {
        var cliVersion = require('../package').version;
        var libVersion = require('cordova-lib/package').version;
        var toPrint = cliVersion;
        if (cliVersion != libVersion || /-dev$/.exec(libVersion)) {
            toPrint += ' (cordova-lib@' + libVersion + ')';
        }
        console.log(toPrint);
        return;
    }


    // For CordovaError print only the message without stack trace unless we
    // are in a verbose mode.
    process.on('uncaughtException', function(err) {
        logger.error(err);
        process.exit(1);
    });

    events.on('verbose', logger.verbose);
    events.on('log', logger.normal);
    events.on('info', logger.info);
    events.on('warn', logger.warn);

    // Set up event handlers for logging and results emitted as events.
    events.on('results', logger.results);

    if (args.silent) {
        logger.setLevel('error');
    }

    if (args.verbose) {
        logger.setLevel('verbose');
    }

    // TODO: Example wanted, is this functionality ever used?
    // If there were arguments protected from nopt with a double dash, keep
    // them in unparsedArgs. For example:
    // cordova build ios -- --verbose --whatever
    // In this case "--verbose" is not parsed by nopt and args.vergbose will be
    // false, the unparsed args after -- are kept in unparsedArgs and can be
    // passed downstream to some scripts invoked by Cordova.
    var unparsedArgs = [];
    var parseStopperIdx =  args.argv.original.indexOf('--');
    if (parseStopperIdx != -1) {
        unparsedArgs = args.argv.original.slice(parseStopperIdx + 1);
    }

    // args.argv.remain contains both the undashed args (like platform names)
    // and whatever unparsed args that were protected by " -- ".
    // "undashed" stores only the undashed args without those after " -- " .
    var remain = args.argv.remain;
    var undashed = remain.slice(0, remain.length - unparsedArgs.length);
    var cmd = undashed[0];
    var subcommand;
    var msg;
    var known_platforms = Object.keys(cordova_lib.cordova_platforms);

    if ( !cmd || cmd == 'help' || args.help ) {
        if (!args.help && remain[0] == 'help') {
            remain.shift();
        }
        return help(remain);
    }

    if ( !cordova.hasOwnProperty(cmd) ) {
        msg =
            'Cordova does not know ' + cmd + '; try `' + cordova_lib.binname +
            ' help` for a list of all the available commands.';
        throw new CordovaError(msg);
    }

    var opts = {
        platforms: [],
        options: [],
        verbose: args.verbose || false,
        silent: args.silent || false,
        browserify: args.browserify || false,
        searchpath : args.searchpath
    };

    if (cmd == 'emulate' || cmd == 'build' || cmd == 'prepare' || cmd == 'compile' || cmd == 'run' || cmd === 'clean') {
        // All options without dashes are assumed to be platform names
        opts.platforms = undashed.slice(1);
        var badPlatforms = _.difference(opts.platforms, known_platforms);
        if( !_.isEmpty(badPlatforms) ) {
            msg = 'Unknown platforms: ' + badPlatforms.join(', ');
            throw new CordovaError(msg);
        }

        // CB-6976 Windows Universal Apps. Allow mixing windows and windows8 aliases
        opts.platforms = opts.platforms.map(function(platform) {
            // allow using old windows8 alias for new unified windows platform
            if (platform == 'windows8' && fs.existsSync('platforms/windows')) {
                return 'windows';
            }
            // allow using new windows alias for old windows8 platform
            if (platform == 'windows' &&
                !fs.existsSync('platforms/windows') &&
                fs.existsSync('platforms/windows8')) {
                return 'windows8';
            }
            return platform;
        });

        // Pass nopt-parsed args to PlatformApi through opts.options
        opts.options = args;
        opts.options.argv = unparsedArgs;

        if (cmd == 'run' && args.list && cordova.raw.targets) {
            cordova.raw.targets.call(null, opts).done();
            return;
        }

        cordova.raw[cmd].call(null, opts).done();
    } else if (cmd === 'requirements') {
        // All options without dashes are assumed to be platform names
        opts.platforms = undashed.slice(1);
        var badPlatforms = _.difference(opts.platforms, known_platforms);
        if( !_.isEmpty(badPlatforms) ) {
            msg = 'Unknown platforms: ' + badPlatforms.join(', ');
            throw new CordovaError(msg);
        }

        // CB-6976 Windows Universal Apps. Allow mixing windows and windows8 aliases
        opts.platforms = opts.platforms.map(function(platform) {
            // allow using old windows8 alias for new unified windows platform
            if (platform == 'windows8' && fs.existsSync('platforms/windows')) {
                return 'windows';
            }
            // allow using new windows alias for old windows8 platform
            if (platform == 'windows' &&
                !fs.existsSync('platforms/windows') &&
                fs.existsSync('platforms/windows8')) {
                return 'windows8';
            }
            return platform;
        });

        cordova.raw[cmd].call(null, opts.platforms)
        .then(function (platformChecks) {

            var someChecksFailed = Object.keys(platformChecks).map(function (platformName) {
                events.emit('log', '\nRequirements check results for ' + platformName + ':');
                var platformCheck = platformChecks[platformName];
                if (platformCheck instanceof CordovaError) {
                    events.emit('warn', 'Check failed for ' + platformName + ' due to ' + platformCheck);
                    return true;
                }

                var someChecksFailed = false;
                platformCheck.forEach(function (checkItem) {
                    var checkSummary = checkItem.name + ': ' +
                                    (checkItem.installed ? 'installed ' : 'not installed ') +
                                    (checkItem.metadata.version || '');
                    events.emit('log', checkSummary);
                    if (!checkItem.installed) {
                        someChecksFailed = true;
                        events.emit('warn', checkItem.metadata.reason);
                    }
                });

                return someChecksFailed;
            }).some(function (isCheckFailedForPlatform) {
                return isCheckFailedForPlatform;
            });

            if (someChecksFailed) throw new CordovaError('Some of requirements check failed');
        }).done();
    } else if (cmd == 'serve') {
        var port = undashed[1];
        cordova.raw.serve(port).done();
    } else if (cmd == 'create') {
        var cfg = {};
        // If we got a fourth parameter, consider it to be JSON to init the config.
        if ( undashed[4] ) {
            cfg = JSON.parse(undashed[4]);
        }
        var customWww = args['copy-from'] || args['link-to'];
        if (customWww) {
            if (customWww.indexOf('http') === 0) {
                throw new CordovaError(
                    'Only local paths for custom www assets are supported.'
                );
            }
            if ( customWww.substr(0,1) === '~' ) {  // resolve tilde in a naive way.
                customWww = path.join(process.env.HOME,  customWww.substr(1));
            }
            customWww = path.resolve(customWww);
            var wwwCfg = { url: customWww };
            if (args['link-to']) {
                wwwCfg.link = true;
            }
            cfg.lib = cfg.lib || {};
            cfg.lib.www = wwwCfg;
        }
        // create(dir, id, name, cfg)
        cordova.raw.create( undashed[1]  // dir to create the project in
                          , undashed[2]  // App id
                          , undashed[3]  // App name
                          , cfg
        ).done();
    } else {
        // platform/plugins add/rm [target(s)]
        subcommand = undashed[1]; // sub-command like "add", "ls", "rm" etc.
        var targets = undashed.slice(2); // array of targets, either platforms or plugins
        var cli_vars = {};
        if (args.variable) {
            args.variable.forEach(function (s) {
                // CB-9171
                var eq = s.indexOf('=');
                if (eq == -1)
                    throw new CordovaError("invalid variable format: " + s);
                var key = s.substr(0, eq).toUpperCase();
                var val = s.substr(eq + 1, s.length);
                cli_vars[key] = val;
            });
        }
        var download_opts = { searchpath : args.searchpath
                            , noregistry : args.noregistry
                            , usegit : args.usegit
                            , cli_variables : cli_vars
                            , browserify: args.browserify || false
                            , link: args.link || false
                            , save: args.save || false
                            , shrinkwrap: args.shrinkwrap || false
                            };
        cordova.raw[cmd](subcommand, targets, download_opts).done();
    }
}
