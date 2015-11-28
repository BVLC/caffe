'use strict';

var fs              = require('fs-extra');
var path            = require('path');
var Builder         = require('../../../lib/models/builder');
var BuildCommand    = require('../../../lib/commands/build');
var commandOptions  = require('../../factories/command-options');
var touch           = require('../../helpers/file-utils').touch;
var existsSync      = require('exists-sync');
var expect          = require('chai').expect;
var Promise         = require('../../../lib/ext/promise');
var stub            = require('../../helpers/stub').stub;
var MockProject     = require('../../helpers/mock-project');
var remove          = Promise.denodeify(fs.remove);
var tmp             = require('tmp-sync');

var root            = process.cwd();
var tmproot         = path.join(root, 'tmp');

describe('models/builder.js', function() {
  var addon, builder, buildResults, outputPath, tmpdir;

  describe('copyToOutputPath', function() {
    beforeEach(function() {
      tmpdir  = tmp.in(tmproot);

      builder = new Builder({
        setupBroccoliBuilder: function() { },
        trapSignals: function() { },
        cleanupOnExit: function() { },
        project: new MockProject()
      });
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('allows for non-existent output-paths at arbitrary depth', function() {
      builder.outputPath = path.join(tmpdir, 'some', 'path', 'that', 'does', 'not', 'exist');

      return builder.copyToOutputPath('tests/fixtures/blueprints/basic_2')
        .then(function() {
          expect(existsSync(path.join(builder.outputPath, 'files', 'foo.txt'))).to.equal(true);
        });
    });
  });

  it('clears the outputPath when multiple files are present', function() {
    outputPath     = 'tmp/builder-fixture/';
    var firstFile  = outputPath + '/assets/foo-bar.js';
    var secondFile = outputPath + '/assets/baz-bif.js';

    fs.mkdirsSync(outputPath + '/assets/');
    touch(firstFile);
    touch(secondFile);

    builder = new Builder({
      setupBroccoliBuilder: function() { },
      trapSignals:          function() { },
      cleanupOnExit:        function() { },

      outputPath: outputPath,
      project: new MockProject()
    });

    return builder.clearOutputPath()
      .then(function() {
        expect(existsSync(firstFile)).to.equal(false);
        expect(existsSync(secondFile)).to.equal(false);
      });
  });

  describe('Prevent deletion of files for improper outputPath', function() {
    var command;
    var parentPath = '..' + path.sep + '..' + path.sep;

    before(function() {
      command = new BuildCommand(commandOptions());

      builder = new Builder({
        setupBroccoliBuilder: function() { },
        trapSignals: function() { },
        cleanupOnExit: function() { },
        project: new MockProject()
      });
    });

    it('when outputPath is root directory ie., `--output-path=/` or `--output-path=C:`', function() {
      var outputPathArg = '--output-path=.';
      var outputPath = command.parseArgs([outputPathArg]).options.outputPath;
      outputPath = outputPath.split(path.sep)[0] + path.sep;
      builder.outputPath = outputPath;

      expect(builder.canDeleteOutputPath(outputPath)).to.equal(false);
    });

    it('when outputPath is project root ie., `--output-path=.`', function() {
      var outputPathArg = '--output-path=.';
      var outputPath = command.parseArgs([outputPathArg]).options.outputPath;
      builder.outputPath = outputPath;

      expect(builder.canDeleteOutputPath(outputPath)).to.equal(false);
    });

    it('when outputPath is a parent directory ie., `--output-path=' + parentPath + '`', function() {
      var outputPathArg = '--output-path=' + parentPath;
      var outputPath = command.parseArgs([outputPathArg]).options.outputPath;
      builder.outputPath = outputPath;

      expect(builder.canDeleteOutputPath(outputPath)).to.equal(false);
    });

    it('allow outputPath to contain the root path as a substring, as long as it is not a parent', function() {
      var outputPathArg = '--output-path=.';
      var outputPath = command.parseArgs([outputPathArg]).options.outputPath;
      outputPath = outputPath.substr(0, outputPath.length - 1);
      builder.outputPath = outputPath;

      expect(builder.canDeleteOutputPath(outputPath)).to.equal(true);
    });
  });

  describe('addons', function() {
    var hooksCalled;

    beforeEach(function() {
      hooksCalled = [];
      addon = {
        name: 'TestAddon',
        preBuild: function() {
          hooksCalled.push('preBuild');

          return Promise.resolve();
        },

        postBuild: function() {
          hooksCalled.push('postBuild');

          return Promise.resolve();
        },

        outputReady: function() {
          hooksCalled.push('outputReady');
        },

        buildError: function() {
          hooksCalled.push('buildError');
        },
      };

      builder = new Builder({
        setupBroccoliBuilder: function() { },
        trapSignals:          function() { },
        cleanupOnExit:        function() { },
        builder: {
          build: function() {
            hooksCalled.push('build');

            return Promise.resolve(buildResults);
          }
        },
        processBuildResult: function(buildResults) { return Promise.resolve(buildResults); },
        project: {
          addons: [addon]
        }
      });

      buildResults = 'build results';
    });

    it('allows addons to add promises preBuild', function() {
      var preBuild = stub(addon, 'preBuild', Promise.resolve());

      return builder.build().then(function() {
        expect(preBuild.called).to.equal(1, 'expected preBuild to be called');
      });
    });

    it('allows addons to add promises postBuild', function() {
      var postBuild = stub(addon, 'postBuild');

      return builder.build().then(function() {
        expect(postBuild.called).to.equal(1, 'expected postBuild to be called');
        expect(postBuild.calledWith[0][0]).to.equal(buildResults, 'expected postBuild to be called with the results');
      });
    });

    it('allows addons to add promises outputReady', function() {
      var outputReady = stub(addon, 'outputReady');

      return builder.build().then(function() {
        expect(outputReady.called).to.equal(1, 'expected outputReady to be called');
        expect(outputReady.calledWith[0][0]).to.equal(buildResults, 'expected outputReady to be called with the results');
      });
    });

    it('hooks are called in the right order', function() {
      return builder.build().then(function() {
        expect(hooksCalled).to.deep.equal(['preBuild', 'build', 'postBuild', 'outputReady']);
      });
    });

    it('should call postBuild before processBuildResult', function() {
      var called = [];

      addon.postBuild = function() {
        called.push('postBuild');
      };

      builder.processBuildResult = function() {
        called.push('processBuildResult');
      };

      return builder.build().then(function() {
        expect(called).to.deep.equal(['postBuild', 'processBuildResult']);
      });
    });

    it('should call outputReady after processBuildResult', function() {
      var called = [];

      builder.processBuildResult = function() {
        called.push('processBuildResult');
      };

      addon.outputReady = function() {
        called.push('outputReady');
      };

      return builder.build().then(function() {
        expect(called).to.deep.equal(['processBuildResult', 'outputReady']);
      });
    });

    it('buildError receives the error object from the errored step', function() {
      var thrownBuildError = new Error('buildError');
      var receivedBuildError;

      addon.buildError = function(errorThrown) {
        receivedBuildError = errorThrown;
      };

      builder.builder.build = function() {
        hooksCalled.push('build');

        return Promise.reject(thrownBuildError);
      };

      return builder.build().then(function() {
        expect(false, 'should not succeed');
      }).catch(function() {
        expect(receivedBuildError).to.equal(thrownBuildError);
      });
    });

    it('calls buildError and does not call build, postBuild or outputReady when preBuild fails', function() {
      addon.preBuild = function() {
        hooksCalled.push('preBuild');

        return Promise.reject(new Error('preBuild Error'));
      };

      return builder.build().then(function() {
        expect(false, 'should not succeed');
      }).catch(function() {
        expect(hooksCalled).to.deep.equal(['preBuild', 'buildError']);
      });
    });

    it('calls buildError and does not call postBuild or outputReady when build fails', function() {
      builder.builder.build = function() {
        hooksCalled.push('build');

        return Promise.reject(new Error('build Error'));
      };

      return builder.build().then(function() {
        expect(false, 'should not succeed');
      }).catch(function() {
        expect(hooksCalled).to.deep.equal(['preBuild', 'build', 'buildError']);
      });
    });

    it('calls buildError when postBuild fails', function() {
      addon.postBuild = function() {
        hooksCalled.push('postBuild');

        return Promise.reject(new Error('preBuild Error'));
      };

      return builder.build().then(function() {
        expect(false, 'should not succeed');
      }).catch(function() {
        expect(hooksCalled).to.deep.equal(['preBuild', 'build', 'postBuild', 'buildError']);
      });
    });

    it('calls buildError when outputReady fails', function() {
      addon.outputReady = function() {
        hooksCalled.push('outputReady');

        return Promise.reject(new Error('outputReady Error'));
      };

      return builder.build().then(function() {
        expect(false, 'should not succeed');
      }).catch(function() {
        expect(hooksCalled).to.deep.equal(['preBuild', 'build', 'postBuild', 'outputReady', 'buildError']);
      });
    });
  });
});
