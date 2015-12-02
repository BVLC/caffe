/*
 * grunt-contrib-bump
 * http://gruntjs.com/
 *
 * Copyright (c) 2014 "Cowboy" Ben Alman, contributors
 * Licensed under the MIT license.
 */

'use strict';

var semver = require('semver');
var shell = require('shelljs');

module.exports = function(grunt) {

  grunt.registerTask('bump', 'Bump the version property of a JSON file.', function() {
    // Validate specified semver increment modes.
    var valids = ['major', 'minor', 'patch', 'prerelease'];
    var modes = [];
    this.args.forEach(function(mode) {
      var matches = [];
      valids.forEach(function(valid) {
        if (valid.indexOf(mode) === 0) { matches.push(valid); }
      });
      if (matches.length === 0) {
        grunt.log.error('Error: mode "' + mode + '" does not match any known modes.');
      } else if (matches.length > 1) {
        grunt.log.error('Error: mode "' + mode + '" is ambiguous (possibly: ' + matches.join(', ') + ').');
      } else {
        modes.push(matches[0]);
      }
    });
    if (this.errorCount === 0 && modes.length === 0) {
      grunt.log.error('Error: no modes specified.');
    }
    if (this.errorCount > 0) {
      grunt.log.error('Valid modes are: ' + valids.join(', ') + '.');
      throw new Error('Use valid modes (or unambiguous mode abbreviations).');
    }
    // Options.
    var options = this.options({
      filepaths: ['package.json'],
      syncVersions: false,
      commit: true,
      commitMessage: 'Bumping version to {%= version %}.',
      tag: true,
      tagName: 'v{%= version %}',
      tagMessage: 'Version {%= version %}',
      tagPrerelease: false,
    });
    // Normalize filepaths to array.
    var filepaths = Array.isArray(options.filepaths) ? options.filepaths : [options.filepaths];
    // Process JSON files, in-order.
    var versions = {};
    filepaths.forEach(function(filepath) {
      var o = grunt.file.readJSON(filepath);
      var origVersion = o.version;
      // If syncVersions is enabled, only grab version from the first file,
      // guaranteeing new versions will always be in sync.
      var firstVersion = Object.keys(versions)[0];
      if (options.syncVersions && firstVersion) {
        o.version = firstVersion;
      }
      modes.forEach(function(mode) {
        var orig = o.version;
        var s = semver.parse(o.version);
        s.inc(mode);
        o.version = String(s);
        // Workaround for https://github.com/isaacs/node-semver/issues/50
        if (/-/.test(orig) && mode === 'patch') {
          o.version = o.version.replace(/\d+$/, function(n) { return n - 1; });
        }
        // If prerelease on an un-prerelease version, bump patch version first
        if (!/-/.test(orig) && mode === 'prerelease') {
          s.inc('patch');
          s.inc('prerelease');
          o.version = String(s);
        }
      });
      if (versions[origVersion]) {
        versions[origVersion].filepaths.push(filepath);
      } else {
        versions[origVersion] = {version: o.version, filepaths: [filepath]};
      }
      // Actually *do* something.
      grunt.log.write('Bumping version in ' + filepath + ' from ' + origVersion + ' to ' + o.version + '...');
      grunt.file.write(filepath, JSON.stringify(o, null, 2));
      grunt.log.ok();
    });
    // Commit changed files?
    if (options.commit) {
      Object.keys(versions).forEach(function(origVersion) {
        var o = versions[origVersion];
        commit(o.filepaths, processTemplate(options.commitMessage, {
          version: o.version,
          origVersion: origVersion
        }));
      });
    }
    // We're only going to create one tag. And it's going to be the new
    // version of the first bumped file. Because, sanity.
    var newVersion = versions[Object.keys(versions)[0]].version;
    if (options.tag) {
      if (options.tagPrerelease || modes.indexOf('prerelease') === -1) {
        tag(
          processTemplate(options.tagName, {version: newVersion}),
          processTemplate(options.tagMessage, {version: newVersion})
        );
      } else {
        grunt.log.writeln('Not tagging (prerelease version).');
      }
    }
    if (this.errorCount > 0) {
      grunt.warn('There were errors.');
    }
  });

  // Using custom delimiters keeps templates from being auto-processed.
  grunt.template.addDelimiters('bump', '{%', '%}');

  function processTemplate(message, data) {
    return grunt.template.process(message, {
      delimiters: 'bump',
      data: data,
    });
  }

  // Kinda borrowed from https://github.com/geddski/grunt-release
  function commit(filepaths, message) {
    grunt.log.writeln('Committing ' + filepaths.join(', ') + ' with message: ' + message);
    run("git commit -m '" + message + "' '" + filepaths.join("' '") + "'");
  }

  function tag(name, message) {
    grunt.log.writeln('Tagging ' + name + ' with message: ' + message);
    run("git tag '" + name + "' -m '" + message + "'");
  }

  function run(cmd) {
    if (grunt.option('no-write')) {
      grunt.verbose.writeln('Not actually running: ' + cmd);
    } else {
      grunt.verbose.writeln('Running: ' + cmd);
      var result = shell.exec(cmd, {silent:true});
      if (result.code !== 0) {
        grunt.log.error('Error (' + result.code + ') ' + result.output);
      }
    }
  }

};