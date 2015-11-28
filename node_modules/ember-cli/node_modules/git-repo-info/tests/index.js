'use strict';

var assert = require('assert');
var path = require('path');
var repoInfo = require('../index');
var zlib = require('zlib');

require('mocha-jshint')();

var root = process.cwd();
var testFixturesPath = path.join(__dirname, 'fixtures');
var gitDir = 'dot-git';

repoInfo._suppressErrors = false;

describe('git-repo-info', function() {
  before(function() {
    repoInfo._changeGitDir(gitDir);
  });

  afterEach(function() {
    process.chdir(root);
  });

  describe('repo lookup', function() {
    var repoRoot = path.join(testFixturesPath, 'nested-repo');

    it('finds a repo in the current directory', function() {
      process.chdir(repoRoot);

      var foundPath = repoInfo._findRepo(repoRoot);
      assert.equal(foundPath, path.join(repoRoot, gitDir));
    });

    it('finds a repo in the parent directory', function() {
      process.chdir(path.join(repoRoot, 'foo'));

      var foundPath = repoInfo._findRepo(repoRoot);
      assert.equal(foundPath, path.join(repoRoot, gitDir));
    });

    it('finds a repo 2 levels up', function() {
      process.chdir(path.join(repoRoot, 'foo', 'bar'));

      var foundPath = repoInfo._findRepo(repoRoot);
      assert.equal(foundPath, path.join(repoRoot, gitDir));
    });

    it('finds a repo without an argument', function() {
      process.chdir(repoRoot);

      var foundPath = repoInfo._findRepo();
      assert.equal(foundPath, path.join(repoRoot, gitDir));
    });
  });

  describe('repoInfo', function() {
    it('returns an object with repo info', function() {
      var repoRoot = path.join(testFixturesPath, 'nested-repo');
      var result = repoInfo(path.join(repoRoot, gitDir));

      var expected = {
        branch: 'master',
        sha: '5359aabd3872d9ffd160712e9615c5592dfe6745',
        abbreviatedSha: '5359aabd38',
        tag: null
      };

      assert.deepEqual(result, expected);
    });

    it('returns an object with repo info', function() {
      var repoRoot = path.join(testFixturesPath, 'detached-head');
      var result = repoInfo(path.join(repoRoot, gitDir));

      var expected = {
        branch: null,
        sha: '9dac893d5a83c02344d91e79dad8904889aeacb1',
        abbreviatedSha: '9dac893d5a',
        tag: null
      };

      assert.deepEqual(result, expected);
    });


    it('returns an object with repo info, including the tag (packed tags)', function() {
      var repoRoot = path.join(testFixturesPath, 'tagged-commit-packed');
      var result = repoInfo(path.join(repoRoot, gitDir));

      var expected = {
        branch: 'master',
        sha: '5359aabd3872d9ffd160712e9615c5592dfe6745',
        abbreviatedSha: '5359aabd38',
        tag: 'my-tag'
      };

      assert.deepEqual(result, expected);
    });

    it('returns an object with repo info, including the tag (unpacked tags)', function() {
      var repoRoot = path.join(testFixturesPath, 'tagged-commit-unpacked');
      var result = repoInfo(path.join(repoRoot, gitDir));

      var expected = {
        branch: 'master',
        sha: 'c1ee41c325d54f410b133e0018c7a6b1316f6cda',
        abbreviatedSha: 'c1ee41c325',
        tag: 'awesome-tag'
      };

      assert.deepEqual(result, expected);
    });

    it('returns an object with repo info, including the tag (unpacked tags) when a tag object does not exist', function() {
      var repoRoot = path.join(testFixturesPath, 'tagged-commit-unpacked-no-object');
      var result = repoInfo(path.join(repoRoot, gitDir));

      var expected = {
        branch: 'master',
        sha: 'c1ee41c325d54f410b133e0018c7a6b1316f6cda',
        abbreviatedSha: 'c1ee41c325',
        tag: 'awesome-tag'
      };

      assert.deepEqual(result, expected);
    });

    if (zlib.inflateSync) {
      it('returns an object with repo info, including the tag (annotated tags)', function() {
        var repoRoot = path.join(testFixturesPath, 'tagged-annotated');
        var result = repoInfo(path.join(repoRoot, gitDir));

        var expected = {
          branch: 'master',
          sha: 'c1ee41c325d54f410b133e0018c7a6b1316f6cda',
          abbreviatedSha: 'c1ee41c325',
          tag: 'awesome-tag'
        };

        assert.deepEqual(result, expected);
      });
    }

    it('returns an object with repo info, including the full branch name, if the branch name includes any slashes', function() {
      var repoRoot = path.join(testFixturesPath, 'branch-with-slashes');
      var result = repoInfo(path.join(repoRoot, gitDir));

      var expected = {
        branch: 'feature/branch/with/slashes',
        sha: '5359aabd3872d9ffd160712e9615c5592dfe6745',
        abbreviatedSha: '5359aabd38',
        tag: null
      };

      assert.deepEqual(result, expected);
    });
  });
});
