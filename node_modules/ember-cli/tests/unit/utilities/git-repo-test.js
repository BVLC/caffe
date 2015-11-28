'use strict';

var isGitRepo = require('is-git-url');
var expect    = require('chai').expect;

describe('cleanBaseURL()', function() {
  it('recognizes git-style urls in various formats', function() {
    expect(isGitRepo('https://github.com/trek/app-blueprint-test.git'));
    expect(isGitRepo('git@github.com:trek/app-blueprint-test.git'));
    expect(isGitRepo('git+ssh://user@server/project.git'));
    expect(isGitRepo('git+https://user@server/project.git'));
  });
});
