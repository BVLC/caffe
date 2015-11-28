'use strict';

var Promise  = require('../../lib/ext/promise');
var Task     = require('../models/task');
var exec     = Promise.denodeify(require('child_process').exec);
var path     = require('path');
var pkg      = require('../../package.json');
var fs       = require('fs');
var template = require('lodash/string/template');

var gitEnvironmentVariables = {
  GIT_AUTHOR_NAME: 'Tomster',
  GIT_AUTHOR_EMAIL: 'tomster@emberjs.com',
  get GIT_COMMITTER_NAME(){ return this.GIT_AUTHOR_NAME; },
  get GIT_COMMITTER_EMAIL(){ return this.GIT_AUTHOR_EMAIL; }
};

module.exports = Task.extend({
  run: function(commandOptions) {
    var chalk  = require('chalk');
    var ui     = this.ui;

    if(commandOptions.skipGit) { return Promise.resolve(); }

    return exec('git --version')
      .then(function() {
        return exec('git init')
          .then(function() {
          return exec('git add .');
        })
        .then(function(){
          var commitTemplate = fs.readFileSync(path.join(__dirname, '../utilities/COMMIT_MESSAGE.txt'));
          var commitMessage = template(commitTemplate)(pkg);
          return exec('git commit -m "' + commitMessage + '"', {env: gitEnvironmentVariables});
        })
        .then(function(){
          ui.writeLine(chalk.green('Successfully initialized git.'));
        });
      }).catch(function(/*error*/){
        // if git is not found or an error was thrown during the `git`
        // init process just swallow any errors here
    });
  }
});
