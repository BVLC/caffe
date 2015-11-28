'use strict';

var Promise    = require('../../lib/ext/promise');
var path       = require('path');
var fs         = require('fs-extra');
var remove     = Promise.denodeify(fs.remove);

var expect     = require('chai').expect;
var EOL        = require('os').EOL;

var runCommand          = require('../helpers/run-command');
var acceptance          = require('../helpers/acceptance');
var copyFixtureFiles    = require('../helpers/copy-fixture-files');
var assertDirEmpty      = require('../helpers/assert-dir-empty');
var existsSync          = require('exists-sync');
var createTestTargets   = acceptance.createTestTargets;
var teardownTestTargets = acceptance.teardownTestTargets;
var linkDependencies    = acceptance.linkDependencies;
var cleanupRun          = acceptance.cleanupRun;
var existsSync          = require('exists-sync');

var appName  = 'some-cool-app';

describe('Acceptance: brocfile-smoke-test', function() {
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

  it('a custom EmberENV in config/environment.js is used for window.EmberENV', function() {
    return copyFixtureFiles('brocfile-tests/custom-ember-env')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var vendorContents = fs.readFileSync(path.join('dist', 'assets', 'vendor.js'), {
          encoding: 'utf8'
        });

        var expected = 'window.EmberENV = {"asdflkmawejf":";jlnu3yr23"};';
        expect(vendorContents).to.contain(expected, 'EmberENV should be in assets/vendor.js');
      });
  });

  it('a custom environment config can be used in Brocfile.js', function() {
    return copyFixtureFiles('brocfile-tests/custom-environment-config')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
      });
  });

  it('using wrapInEval: true', function() {
    return copyFixtureFiles('brocfile-tests/wrap-in-eval')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
      });
  });

  it('without app/templates', function() {
    return copyFixtureFiles('brocfile-tests/pods-templates')
      .then(function(){
        // remove ./app/templates
        return remove(path.join(process.cwd(), 'app/templates'));
      }).then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
      });
  });

  it('strips app/styles or app/templates from JS', function() {
    return copyFixtureFiles('brocfile-tests/styles-and-templates-stripped')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var appFileContents = fs.readFileSync(path.join('.', 'dist', 'assets', appName + '.js'), {
          encoding: 'utf8'
        });

        expect(appFileContents).to.include('//app/templates-stuff.js');
        expect(appFileContents).to.include('//app/styles-manager.js');
      });
  });

  it('should fall back to the Brocfile', function() {
    return copyFixtureFiles('brocfile-tests/no-ember-cli-build').then(function() {
      fs.removeSync('./ember-cli-build.js');
      return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
    }).then(function() {
      expect(existsSync(path.join('.', 'Brocfile.js'))).to.be.ok;
      expect(existsSync(path.join('.', 'ember-cli-build.js'))).to.be.not.ok;
    });
  });

  it('should use the Brocfile if both a Brocfile and ember-cli-build exist', function() {
    return copyFixtureFiles('brocfile-tests/both-build-files').then(function() {
      return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
    }).then(function(result) {
      var vendorContents = fs.readFileSync(path.join('dist', 'assets', 'vendor.js'), {
        encoding: 'utf8'
      });

      var expected = 'var usingBrocfile = true;';

      expect(vendorContents).to.contain(expected, 'includes file imported from Brocfile');
      expect(result.output.join('\n')).to.include('Brocfile.js has been deprecated');
    });
  });

  it('should throw if no build file is found', function() {
    fs.removeSync('./ember-cli-build.js');
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build').catch(function(err) {
      expect(err.code).to.eql(1);
    });
  });

  it('using autoRun: true', function() {
    return copyFixtureFiles('brocfile-tests/auto-run-true')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var appFileContents = fs.readFileSync(path.join('.', 'dist', 'assets', appName + '.js'), {
          encoding: 'utf8'
        });

        expect(appFileContents).to.match(/\/app"\)\["default"\]\.create\(/);
      });
  });

  it('using autoRun: false', function() {

    return copyFixtureFiles('brocfile-tests/auto-run-false')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var appFileContents = fs.readFileSync(path.join('.', 'dist', 'assets', appName + '.js'), {
          encoding: 'utf8'
        });

        expect(appFileContents).to.not.match(/\/app"\)\["default"\]\.create\(/);
      });
  });

  it('default development build does not fail', function() {
    return copyFixtureFiles('brocfile-tests/query')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      });
  });

  it('default development build tests', function() {
    return copyFixtureFiles('brocfile-tests/default-development')
    .then(function() {
      return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
    });
  });

  it('app.import works properly with test tree files', function() {
    return copyFixtureFiles('brocfile-tests/app-test-import')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = JSON.parse(fs.readFileSync(packageJsonPath,'utf8'));
        packageJson.devDependencies['ember-test-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var subjectFileContents = fs.readFileSync(path.join('.', 'dist', 'assets', 'test-support.js'), {
          encoding: 'utf8'
        });

        expect(subjectFileContents.indexOf('// File for test tree imported and added via postprocessTree()') > 0).to.equal(true);
      });
  });

  it('app.import works properly with non-js/css files', function() {
    return copyFixtureFiles('brocfile-tests/app-import')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = JSON.parse(fs.readFileSync(packageJsonPath,'utf8'));
        packageJson.devDependencies['ember-random-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var subjectFileContents = fs.readFileSync(path.join('.', 'dist', 'assets', 'file-to-import.txt'), {
          encoding: 'utf8'
        });

        expect(subjectFileContents).to.equal('EXAMPLE TEXT FILE CONTENT' + EOL);
      });
  });

  it('app.import fails when options.type is not `vendor` or `test`', function(){
    return copyFixtureFiles('brocfile-tests/app-import')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = JSON.parse(fs.readFileSync(packageJsonPath,'utf8'));
        packageJson.devDependencies['ember-bad-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        expect(false, 'Build passed when it should have failed!');
      }, function() {
        expect(true, 'Build failed with invalid options type.');
      });
  });

  it('addons can have a public tree that is merged and returned namespaced by default', function() {
    return copyFixtureFiles('brocfile-tests/public-tree')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = JSON.parse(fs.readFileSync(packageJsonPath,'utf8'));
        packageJson.devDependencies['ember-random-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var subjectFileContents = fs.readFileSync(path.join('.', 'dist', 'ember-random-addon', 'some-root-file.txt'), {
          encoding: 'utf8'
        });

        expect(subjectFileContents).to.equal('ROOT FILE' + EOL);
      });
  });

  it('using pods based templates', function() {
    return copyFixtureFiles('brocfile-tests/pods-templates')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
      });
  });

  it('using pods based templates with a podModulePrefix', function() {
    return copyFixtureFiles('brocfile-tests/pods-with-prefix-templates')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test');
      });
  });

  it('addon trees are not jshinted', function() {
    return copyFixtureFiles('brocfile-tests/jshint-addon')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = JSON.parse(fs.readFileSync(packageJsonPath,'utf8'));
        packageJson['ember-addon'] = {
          paths: ['./lib/ember-random-thing']
        };

        fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));

        var badContent = 'var blah = ""' + EOL + 'export default Blah;';
        var appPath = path.join('.', 'lib', 'ember-random-thing', 'app',
                                          'routes', 'horrible-route.js');
        var testSupportPath = path.join('.', 'lib', 'ember-random-thing', 'test-support',
                                          'unit', 'routes', 'horrible-route-test.js');

        fs.writeFileSync(appPath, badContent);
        fs.writeFileSync(testSupportPath, badContent);
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'test', '--filter=jshint');
      });
  });

  it('specifying custom output paths works properly', function() {
    return copyFixtureFiles('brocfile-tests/custom-output-paths')
      .then(function () {
        var themeCSSPath = path.join(__dirname, '..', '..', 'tmp', appName, 'app', 'styles', 'theme.css');
        return fs.writeFileSync(themeCSSPath, 'html, body { margin: 20%; }');
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var files = [
          '/css/app.css',
          '/css/theme/a.css',
          '/js/app.js',
          '/css/vendor.css',
          '/js/vendor.js',
          '/css/test-support.css',
          '/js/test-support.js',
          '/my-app.html'
        ];

        var basePath = path.join('.', 'dist');
        files.forEach(function(file) {
          expect(existsSync(path.join(basePath, file)), file + ' exists');
        });
      });
  });

  it('multiple css files in app/styles/ are output when a preprocessor is not used', function() {
    return copyFixtureFiles('brocfile-tests/multiple-css-files')
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var files = [
          '/assets/some-cool-app.css',
          '/assets/other.css'
        ];

        var basePath = path.join('.', 'dist');
        files.forEach(function(file) {
          expect(existsSync(path.join(basePath, file)), file + ' exists');
        });
      });
  });

  it('specifying partial `outputPaths` hash deep merges options correctly', function() {
    return copyFixtureFiles('brocfile-tests/custom-output-paths')
      .then(function () {

        var themeCSSPath = path.join(__dirname, '..', '..', 'tmp', appName, 'app', 'styles', 'theme.css');
        fs.writeFileSync(themeCSSPath, 'html, body { margin: 20%; }');

        var brocfilePath = path.join(__dirname, '..', '..', 'tmp', appName, 'ember-cli-build.js');
        var brocfile = fs.readFileSync(brocfilePath, 'utf8');

        // remove outputPaths.app.js option
        brocfile = brocfile.replace(/js: '\/js\/app.js'/, '');
        // remove outputPaths.app.css.app option
        brocfile = brocfile.replace(/'app': '\/css\/app\.css',/, '');

        fs.writeFileSync(brocfilePath, brocfile, 'utf8');
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var files = [
          '/css/theme/a.css',
          '/assets/some-cool-app.js',
          '/css/vendor.css',
          '/js/vendor.js',
          '/css/test-support.css',
          '/js/test-support.js'
        ];

        var basePath = path.join('.', 'dist');
        files.forEach(function(file) {
          expect(existsSync(path.join(basePath, file)), file + ' exists');
        });

        expect(!existsSync(path.join(basePath, '/assets/some-cool-app.css')), 'default app.css should not exist');
      });
  });

  it('multiple paths can be CSS preprocessed', function() {
    return copyFixtureFiles('brocfile-tests/multiple-sass-files')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['broccoli-sass'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var mainCSS = fs.readFileSync(path.join('.', 'dist', 'assets', 'main.css'), {
          encoding: 'utf8'
        });
        var themeCSS = fs.readFileSync(path.join('.', 'dist', 'assets', 'theme', 'a.css'), {
          encoding: 'utf8'
        });

        expect(mainCSS).to.equal('body { background: black; }' + EOL, 'main.css contains correct content');
        expect(themeCSS).to.equal('.theme { color: red; }' + EOL, 'theme/a.css contains correct content');
      });
  });

  it('app.css is output to <app name>.css by default', function() {
    return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build')
      .then(function() {
        var exists = existsSync(path.join('.', 'dist', 'assets', appName + '.css'));

        expect(exists, appName + '.css exists');
      });
  });

  // for backwards compat.
  it('app.scss is output to <app name>.css by default', function() {
    return copyFixtureFiles('brocfile-tests/multiple-sass-files')
      .then(function() {
        var brocfilePath = path.join(__dirname, '..', '..', 'tmp', appName, 'ember-cli-build.js');
        var brocfile = fs.readFileSync(brocfilePath, 'utf8');

        // remove custom preprocessCss paths, use app.scss instead
        brocfile = brocfile.replace(/outputPaths.*/, '');

        fs.writeFileSync(brocfilePath, brocfile, 'utf8');

        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['broccoli-sass'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var mainCSS = fs.readFileSync(path.join('.', 'dist', 'assets', appName + '.css'), {
          encoding: 'utf8'
        });

        expect(mainCSS).to.equal('body { background: green; }' + EOL, appName + '.css contains correct content');
      });
  });
});
