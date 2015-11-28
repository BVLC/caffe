'use strict';

var Blueprint     = require('../models/blueprint');
var Task          = require('../models/task');
var Promise       = require('../ext/promise');
var isGitRepo     = require('is-git-url');
var temp          = require('temp');
var childProcess  = require('child_process');
var path          = require('path');

// Automatically track and cleanup temp files at exit
temp.track();

var mkdir = Promise.denodeify(temp.mkdir);
var exec = Promise.denodeify(childProcess.exec);

module.exports = Task.extend({
  run: function(options) {
    var cwd             = process.cwd();
    var name            = options.rawName;
    var blueprintOption = options.blueprint;
    // If we're in a dry run, pretend we changed directories.
    // Pretending we cd'd avoids prompts in the actual current directory.
    var fakeCwd         = path.join(cwd, name);
    var target          = options.dryRun ? fakeCwd : cwd;

    var installOptions = {
      target: target,
      entity: { name: name },
      ui: this.ui,
      analytics: this.analytics,
      project: this.project,
      dryRun: options.dryRun,
      targetFiles: options.targetFiles,
      rawArgs: options.rawArgs
    };

    if (isGitRepo(blueprintOption)) {
      return mkdir('ember-cli').then(function(pathName){
        var execArgs = ['git', 'clone', blueprintOption, pathName].join(' ');
        return exec(execArgs).then(function(){
          return exec('npm install', {cwd: pathName}).then(function(){
            var blueprint = Blueprint.load(pathName);
            return blueprint.install(installOptions);
          });
        });
      });
    } else {
      var blueprintName = blueprintOption || 'app';
      var blueprint = Blueprint.lookup(blueprintName, {
        paths: this.project.blueprintLookupPaths()
      });
      return blueprint.install(installOptions);
    }
  }
});
