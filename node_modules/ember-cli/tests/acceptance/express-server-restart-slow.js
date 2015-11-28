'use strict';

var path                = require('path');
var expect              = require('chai').expect;
var fs                  = require('fs-extra');
var EOL                 = require('os').EOL;
var Promise             = require('../../lib/ext/promise');
var acceptance          = require('../helpers/acceptance');
var runCommand          = require('../helpers/run-command');
var remove              = Promise.denodeify(fs.remove);
var createTestTargets   = acceptance.createTestTargets;
var teardownTestTargets = acceptance.teardownTestTargets;
var linkDependencies    = acceptance.linkDependencies;
var cleanupRun          = acceptance.cleanupRun;


var copyFixtureFiles = require('../helpers/copy-fixture-files');
var assertDirEmpty   = require('../helpers/assert-dir-empty');

// skipped because brittle. needs some TLC
describe.skip('Acceptance: express server restart', function () {
  var appName = 'express-server-restart-test-app';

  before(function() {
    this.timeout(360000);

    return createTestTargets(appName).then(function() {
      process.chdir(appName);
      return copyFixtureFiles('restart-express-server/app-root');
    });
  });

  after(function() {
    this.timeout(15000);
    return teardownTestTargets();
  });

  beforeEach(function() {
    this.timeout(15000);
    return linkDependencies(appName);
  });

  afterEach(function() {
    this.timeout(15000);
    return cleanupRun().then(function() {
      assertDirEmpty('tmp');
    });
  });

  function getRunCommandOptions(onChildSpawned) {
    return {
      onChildSpawned: onChildSpawned,
      killAfterChildSpawnedPromiseResolution: true
    };
  }

  var initialRoot = process.cwd();
  function ensureTestFileContents(expectedContents, message) {
    var contents = fs.readFileSync(path.join(initialRoot, 'tmp', appName, 'foo.txt'), { encoding: 'utf8' });
    expect(contents).to.equal(expectedContents, message);
  }

  function onChildSpawnedSingleCopy(copySrc, expectedContents) {
    return function() {
      process.chdir('server');
      return delay(6000)
        .then(function() {
          ensureTestFileContents('Initial contents of A.', 'Test file has correct contents after initial server start.');
          return copyFixtureFiles(path.join('restart-express-server', copySrc));
        }).then(function() {
          return delay(4000);
        }).then(function() {
          ensureTestFileContents(expectedContents, 'Test file has correct contents after first copy.');
        });
    };
  }

  function onChildSpawnedMultipleCopies() {
    return function() {
      process.chdir('server');
      return delay(6000)
        .then(function() {
          ensureTestFileContents('Initial contents of A.', 'Test file has correct contents after initial server start.');
          return copyFixtureFiles(path.join('restart-express-server', 'copy1'));
        }).then(function() {
          return delay(4000);
        }).then(function() {
          ensureTestFileContents('Copy1 contents of A.', 'Test file has correct contents after first copy.');
          return copyFixtureFiles(path.join('restart-express-server', 'copy2'));
        }).then(function() {
          return delay(4000);
        }).then(function() {
          ensureTestFileContents('Copy2 contents of A. Copy2 contents of B.', 'Test file has correct contents after second copy.');
          return remove(path.join('restart-express-server', 'subfolder'));
        }).then(function() {
          return copyFixtureFiles(path.join('restart-express-server', 'copy3'));
        }).then(function() {
          return delay(4000);
        }).then(function() {
          ensureTestFileContents('true true', 'Test file has correct contents after second copy.');
        });
    };
  }

  function runServer(commandOptions) {
    return new Promise(function(resolve, reject) {
      return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'),
        'serve',
        '--live-reload-port', '32580',
        '--port', '49741', commandOptions)
        .then(function() {
          throw new Error('The server should not have exited successfully.');
        })
        .catch(function(err) {
          if (err.testingError) {
            return reject(err.testingError);
          }

          // This error was just caused by us having to kill the program
          return resolve();
        });
    });
  }

  it('Server restarts successfully on copy1', function() {
    this.timeout(30000);

    ensureTestFileContents('Initial Contents' + EOL, 'Test file initialized properly.');
    return runServer(getRunCommandOptions(onChildSpawnedSingleCopy('copy1', 'Copy1 contents of A.')));
  });

  it('Server restarts successfully on copy2', function() {
    this.timeout(30000);

    ensureTestFileContents('Initial Contents' + EOL, 'Test file initialized properly.');
    return runServer(getRunCommandOptions(onChildSpawnedSingleCopy('copy2', 'Copy2 contents of A. Copy2 contents of B.')));
  });

  it('Server restarts successfully on multiple copies', function() {
    this.timeout(90000);

    ensureTestFileContents('Initial Contents' + EOL, 'Test file initialized properly.');
    return runServer(getRunCommandOptions(onChildSpawnedMultipleCopies()));
  });
});

function delay(ms) {
  return new Promise(function (resolve) {
    setTimeout(resolve, ms);
  });
}
