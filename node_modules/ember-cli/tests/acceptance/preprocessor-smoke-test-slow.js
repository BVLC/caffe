'use strict';

var path       = require('path');
var fs         = require('fs');
var expect     = require('chai').expect;

var runCommand          = require('../helpers/run-command');
var acceptance          = require('../helpers/acceptance');
var copyFixtureFiles    = require('../helpers/copy-fixture-files');
var assertDirEmpty      = require('../helpers/assert-dir-empty');
var createTestTargets   = acceptance.createTestTargets;
var teardownTestTargets = acceptance.teardownTestTargets;
var linkDependencies    = acceptance.linkDependencies;
var cleanupRun          = acceptance.cleanupRun;

var appName  = 'some-cool-app';

describe('Acceptance: preprocessor-smoke-test', function() {
  before(function() {
    this.timeout(360000);
    return createTestTargets(appName);
  });

  after(function() {
    this.timeout(15000);
    return teardownTestTargets();
  });

  beforeEach(function() {
    this.timeout(360000);
    return linkDependencies(appName);
  });

  afterEach(function() {
    this.timeout(15000);
    return cleanupRun().then(function() {
      assertDirEmpty('tmp');
    });
  });

  it('addons with standard preprocessors compile correctly', function() {
    this.timeout(100000);

    return copyFixtureFiles('preprocessor-tests/app-with-addon-with-preprocessors')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['broccoli-sass'] = 'latest';
        packageJson.devDependencies['ember-cool-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var mainCSS = fs.readFileSync(path.join('.', 'dist', 'assets', 'some-cool-app.css'), {
          encoding: 'utf8'
        });

        var vendorCSS = fs.readFileSync(path.join('.', 'dist', 'assets', 'vendor.css'), {
          encoding: 'utf8'
        });

        expect(mainCSS).to.contain('app styles included');
        expect(vendorCSS).to.contain('addon styles included');
      });
  });

  it('addon registry entries are added in the proper order', function() {
    this.timeout(100000);

    return copyFixtureFiles('preprocessor-tests/app-registry-ordering')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['first-dummy-preprocessor'] = 'latest';
        packageJson.devDependencies['second-dummy-preprocessor'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var appJs = fs.readFileSync(path.join('.', 'dist', 'assets', 'some-cool-app.js'), {
          encoding: 'utf8'
        });

        expect(appJs).to.not.contain('__SECOND_PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should not be contained');
        expect(appJs).to.not.contain('__FIRST_PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should not be contained');
        expect(appJs).to.contain('replacedByPreprocessor', 'token should have been replaced in app bundle');
      });
  });

  it('addons without preprocessors compile correctly', function() {
    this.timeout(100000);

    return copyFixtureFiles('preprocessor-tests/app-with-addon-without-preprocessors')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['broccoli-sass'] = 'latest';
        packageJson.devDependencies['ember-cool-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var mainCSS = fs.readFileSync(path.join('.', 'dist', 'assets', 'some-cool-app.css'), {
          encoding: 'utf8'
        });

        var vendorCSS = fs.readFileSync(path.join('.', 'dist', 'assets', 'vendor.css'), {
          encoding: 'utf8'
        });

        expect(mainCSS).to.contain('app styles included');
        expect(vendorCSS).to.contain('addon styles included');
      });
  });

  /*
    [ app ]  -> [ addon ] -> [ preprocessor addon ]
      |             |
      |             |--- preprocessor applies to this
      |
      |-- preprocessor should not apply to this
  */
  it('addons depending on preprocessor addon preprocesses addon but not app', function() {
    this.timeout(100000);

    return copyFixtureFiles('preprocessor-tests/app-with-addon-with-preprocessors-2')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['ember-cool-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var appJs = fs.readFileSync(path.join('.', 'dist', 'assets', 'some-cool-app.js'), {
          encoding: 'utf8'
        });

        var vendorJs = fs.readFileSync(path.join('.', 'dist', 'assets', 'vendor.js'), {
          encoding: 'utf8'
        });

        expect(appJs).to.contain('__PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should not have been replaced in app bundle');
        expect(appJs).to.not.contain('replacedByPreprocessor', 'token should not have been replaced in app bundle');
        expect(vendorJs).to.not.contain('__PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should have been replaced in vendor bundle');
        expect(vendorJs).to.contain('replacedByPreprocessor', 'token should have been replaced in vendor bundle');
      });
  });

  /*
    [ app ]  -> [ addon ] ->  [ addon ] -> [ preprocessor addon ]
      |             |             |
      |             |             |--- preprocessor applies to this
      |             |
      |             |-- preprocessor should not apply to this
      |
      |-- preprocessor should not apply to this
  */
  it('addon N levels deep depending on preprocessor preprocesses that parent addon only', function() {
    this.timeout(100000);

    return copyFixtureFiles('preprocessor-tests/app-with-addon-with-preprocessors-3')
      .then(function() {
        var packageJsonPath = path.join(__dirname, '..', '..', 'tmp', appName, 'package.json');
        var packageJson = require(packageJsonPath);
        packageJson.devDependencies['ember-shallow-addon'] = 'latest';

        return fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson));
      })
      .then(function() {
        return runCommand(path.join('.', 'node_modules', 'ember-cli', 'bin', 'ember'), 'build');
      })
      .then(function() {
        var appJs = fs.readFileSync(path.join('.', 'dist', 'assets', 'some-cool-app.js'), {
          encoding: 'utf8'
        });

        var vendorJs = fs.readFileSync(path.join('.', 'dist', 'assets', 'vendor.js'), {
          encoding: 'utf8'
        });

        expect(appJs).to.contain('__PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should not have been replaced in app bundle');
        expect(appJs).to.not.contain('replacedByPreprocessor', 'token should not have been replaced in app bundle');
        expect(vendorJs).to.not.contain('deep: __PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should have been replaced in deep component');
        expect(vendorJs).to.contain('deep: "replacedByPreprocessor"', 'token should have been replaced in deep component');
        expect(vendorJs).to.contain('shallow: __PREPROCESSOR_REPLACEMENT_TOKEN__', 'token should not have been replaced in shallow component');
        expect(vendorJs).to.not.contain('shallow: "replacedByPreprocessor"', 'token should not have been replaced in shallow component');
      });
  });

});
