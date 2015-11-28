'use strict';

var path     = require('path');
var fs       = require('fs');
var crypto   = require('crypto');
var expect   = require('chai').expect;
var walkSync = require('walk-sync');
var appName  = 'some-cool-app';
var EOL      = require('os').EOL;

var runCommand          = require('../helpers/run-command');
var acceptance          = require('../helpers/acceptance');
var copyFixtureFiles    = require('../helpers/copy-fixture-files');
var killCliProcess      = require('../helpers/kill-cli-process');
var assertDirEmpty      = require('../helpers/assert-dir-empty');
var ember               = require('../helpers/ember');
var createTestTargets   = acceptance.createTestTargets;
var teardownTestTargets = acceptance.teardownTestTargets;
var linkDependencies    = acceptance.linkDependencies;
var cleanupRun          = acceptance.cleanupRun;

describe('Acceptance: smoke-test', function() {
  this.timeout(500000);
  before(function() {
    return createTestTargets(appName);
  });

  after(function() {
    return teardownTestTargets();
  });

  beforeEach(function() {
    return linkDependencies(appName);
  });

  afterEach(function() {
    return cleanupRun().then(function() {
      assertDirEmpty('tmp');
    });
  });

  it('ember new foo, clean from scratch', function() {
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
  });

  it('ember test exits with non-zero when tests fail', function() {
    return copyFixtureFiles('smoke-tests/failing-test')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test')
          .then(function() {
            expect(false, 'should have rejected with a failing test');
          })
          .catch(function(result) {
            expect(result.code).to.equal(1);
          });
      });
  });

  it('ember test exits with non-zero when build fails', function() {
    return copyFixtureFiles('smoke-tests/test-with-syntax-error')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test')
          .then(function() {
            expect(false, 'should have rejected with a failing test');
          })
          .catch(function(result) {
            expect(result.code).to.equal(1);
          });
      });
  });

  it('ember test exits with non-zero when no tests are run', function() {
    return copyFixtureFiles('smoke-tests/no-testem-launchers')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test')
          .then(function() {
            expect(false, 'should have rejected with a failing test');
          })
          .catch(function(result) {
            expect(result.code).to.equal(1);
          });
      });
  });

  // TODO: re-enable, something is funky with test cleanup...
  // it('ember test exits with zero when tests pass', function() {
  //   return copyFixtureFiles('smoke-tests/passing-test')
  //     .then(function() {
  //       return ember(['test'])
  //         .then(function(result) {
  //           expect(result.code).to.equal(0);
  //         })
  //         .catch(function() {
  //           expect(false, 'should NOT have rejected with a failing test');
  //         });
  //     });
  // });

  it('ember test still runs when only a JavaScript testem config exists', function() {
    return copyFixtureFiles('smoke-tests/js-testem-config')
      .then(function() {
        return ember(['test']);
      });
  });

  // there is a bug in here when running the entire suite on Travis
  // when run in isolation, it passes
  // here is the error:
  // test-support-80f2fe63fae0c44478fe0f8af73200a7.js contains the fingerprint (2871106928f813936fdd64f4d16005ac): expected 'test-support-80f2fe63fae0c44478fe0f8af73200a7.js' to include '2871106928f813936fdd64f4d16005ac'
  it.skip('ember new foo, build production and verify fingerprint', function() {
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build', '--environment=production')
      .then(function() {
        var dirPath = path.join('.', 'dist', 'assets');
        var dir = fs.readdirSync(dirPath);
        var files = [];

        dir.forEach(function (filepath) {
          if (filepath === '.gitkeep') {
            return;
          }

          files.push(filepath);

          var file = fs.readFileSync(path.join(dirPath, filepath), { encoding: null });

          var md5 = crypto.createHash('md5');
          md5.update(file);
          var hex = md5.digest('hex');

          expect(filepath).to.contain(hex, filepath + ' contains the fingerprint (' + hex + ')');
        });

        var indexHtml = fs.readFileSync(path.join('.', 'dist', 'index.html'), { encoding: 'utf8' });

        files.forEach(function (filename) {
          expect(indexHtml).to.contain(filename);
        });
      });
  });


  // TODO: restore, test harness npm appears to incorrectly dedupe broccoli-filter, causing this test to fail.
  // manually testing that case, it seems to work correctly, will restore soon.
  it.skip('ember test --environment=production', function() {
    return copyFixtureFiles('smoke-tests/passing-test')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test', '--environment=production');
      })
      .then(function(result) {
        var exitCode = result.code;
        var output = result.output.join(EOL);

        expect(exitCode).to.equal(0, 'exit code should be 0 for passing tests');
        expect(output).to.match(/JSHint/, 'JSHint should be run on production assets');
        expect(output).to.match(/fail\s+0/, 'no failures');
        expect(output).to.match(/pass\s+\d+/, 'man=y passing');
      });
  });

  it('ember test --path with previous build', function() {
    var originalWrite = process.stdout.write;
    var output = [];

    return copyFixtureFiles('smoke-tests/passing-test')
      .then(function() {
        // TODO: Change to using ember() helper once it properly saves build artifacts
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        // TODO: Figure out how to get this to write into the MockUI
        process.stdout.write = (function() {
          return function() {
            output.push(arguments[0]);
          };
        }(originalWrite));
        return ember([ 'test', '--path=dist' ]);
      })
      .then(function(result) {
        expect(result.exitCode).to.equal(0, 'exit code should be 0 for passing tests');

        output = output.join(EOL);
        expect(output).to.match(/JSHint/, 'JSHint should be run');
        expect(output).to.match(/fail\s+0/, 'no failures');
        expect(output).to.match(/pass\s+8/, '1 passing');

        process.stdout.write = originalWrite;
      });
  });

  it('ember new foo, build development, and verify generated files', function() {
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build')
      .then(function() {
        var dirPath = path.join('.', 'dist');
        var paths = walkSync(dirPath);

        expect(paths).to.have.length.below(23, 'expected fewer than 23 files in dist, found ' + paths.length);
      });
  });

  it('ember build exits with non-zero code when build fails', function () {
    var appJsPath   = path.join('.', 'app', 'app.js');
    var ouputContainsBuildFailed = false;

    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build')
      .then(function (result) {
        expect(result.code).to.equal(0, 'expected exit code to be zero, but got ' + result.code);

        // add something broken to the project to make build fail
        fs.appendFileSync(appJsPath, '{(syntaxError>$@}{');

        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build', {
          onOutput: function(string) {
            // discard output as there will be a lot of errors and a long stacktrace
            // just mark that the output contains expected text
            if (!ouputContainsBuildFailed && string.match(/Build failed/)) {
              ouputContainsBuildFailed = true;
            }
          }
        });

      }).then(function () {
        expect(false, 'should have rejected with a failing build');
      }).catch(function (result) {
        expect(ouputContainsBuildFailed, 'command output must contain "Build failed" text');
        expect(result.code).to.not.equal(0, 'expected exit code to be non-zero, but got ' + result.code);
      });
  });

  it('ember new foo, build --watch development, and verify rebuilt after change', function() {
    var touched     = false;
    var appJsPath   = path.join('.', 'app', 'app.js');
    var builtJsPath = path.join('.', 'dist', 'assets', 'some-cool-app.js');
    var text        = 'anotuhaonteuhanothunaothanoteh';
    var line        = 'console.log("' + text + '");';

    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build', '--watch', {
        onOutput: function(string, child) {
          if (touched) {
            if (string.match(/Build successful/)) {
              // build after change to app.js
              var contents  = fs.readFileSync(builtJsPath).toString();
              expect(contents).to.contain(text, 'must contain changed line after rebuild');
              killCliProcess(child);
            }
          } else {
            if (string.match(/Build successful/)) {
              // first build
              touched = true;
              fs.appendFileSync(appJsPath, line);
            }
          }
        }
      })
      .catch(function() {
        // swallowing because of SIGINT
      });
  });

  it('ember new foo, build --watch development, and verify rebuilt after multiple changes', function() {
    var buildCount  = 0;
    var touched     = false;
    var appJsPath   = path.join('.', 'app', 'app.js');
    var builtJsPath = path.join('.', 'dist', 'assets', 'some-cool-app.js');
    var firstText   = 'anotuhaonteuhanothunaothanoteh';
    var firstLine   = 'console.log("' + firstText + '");';
    var secondText  = 'aahsldfjlwioruoiiononociwewqwr';
    var secondLine  = 'console.log("' + secondText + '");';

    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build', '--watch', {
        onOutput: function(string, child) {
          if (buildCount === 0) {
            if (string.match(/Build successful/)) {
              // first build
              touched = true;
              buildCount = 1;
              fs.appendFileSync(appJsPath, firstLine);
            }
          } else if (buildCount === 1) {
            if (string.match(/Build successful/)) {
              // second build
              touched = true;
              buildCount = 2;
              fs.appendFileSync(appJsPath, secondLine);
            }
          } else if (touched && buildCount === 2) {
            if (string.match(/Build successful/)) {
              // build after change to app.js
              var contents  = fs.readFileSync(builtJsPath).toString();
              expect(contents.indexOf(secondText) > 1, 'must contain second changed line after rebuild');
              killCliProcess(child);
            }
          }
        }
      })
      .catch(function() {
        // swallowing because of SIGINT
      });
  });

  it('ember new foo, server, SIGINT clears tmp/', function() {
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'server', '--port=54323','--live-reload=false', {
        onOutput: function(string, child) {
          if (string.match(/Build successful/)) {
            killCliProcess(child);
          }
        }
      })
      .catch(function() {
        // just eat the rejection as we are testing what happens
      });
  });

  it('ember new foo, build production and verify css files are concatenated', function() {
    return copyFixtureFiles('with-styles')
      .then(function() {
      return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build', '--environment=production')
        .then(function() {
          var dirPath = path.join('.', 'dist', 'assets');
          var dir = fs.readdirSync(dirPath);
          var cssNameRE = new RegExp(appName + '-([a-f0-9]+)\\.css','i');
          dir.forEach(function (filepath) {
            if(cssNameRE.test(filepath)) {
              var appCss = fs.readFileSync(path.join('.', 'dist', 'assets', filepath), { encoding: 'utf8' });
              expect(appCss).to.contain('.some-weird-selector');
              expect(appCss).to.contain('.some-even-weirder-selector');
            }
          });
        });
    });
  });

  it('ember new foo, build production and verify single "use strict";', function() {
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build', '--environment=production')
      .then(function() {
          var dirPath = path.join('.', 'dist', 'assets');
          var dir = fs.readdirSync(dirPath);
          var appNameRE = new RegExp(appName + '-([a-f0-9]+)\\.js','i');
          dir.forEach(function(filepath) {
            if (appNameRE.test(filepath)) {
              var contents = fs.readFileSync(path.join('.', 'dist', 'assets', filepath), { encoding: 'utf8' });
              var count = (contents.match(/(["'])use strict\1;/g) || []).length;
              expect(count).to.equal(1);
            }
          });
      });
  });

  it('ember can override and reuse the built-in blueprints', function() {
    return copyFixtureFiles('addon/with-blueprint-override')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'generate', 'component', 'foo-bar', '-p');
      })
      .then(function() {
        // because we're overriding, the fileMapTokens is default, sans 'component'
        var componentPath = path.join('app','foo-bar','component.js');
        var contents = fs.readFileSync(componentPath, { encoding: 'utf8' });

        expect(contents).to.contain('generated component successfully');
      });
  });
});
