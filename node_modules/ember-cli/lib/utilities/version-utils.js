'use strict';

var existsSync  = require('exists-sync');
var path        = require('path');
var getRepoInfo = require('git-repo-info');

module.exports = {
  emberCLIVersion: function emberCLIVersion() {
    var gitPath = path.join(__dirname, '..','..','.git');
    var output  = [require('../../package.json').version];

    if (existsSync(gitPath)) {
      var repoInfo = getRepoInfo(gitPath);

      output.push(repoInfo.branch);
      output.push(repoInfo.abbreviatedSha);
    }

    return output.join('-');
  },

  isDevelopment: function isDevelopment(version) {
    // match postfix SHA in dev version
    return !!version.match(/\b[0-9a-f]{5,40}\b/);
  }
};
