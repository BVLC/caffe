'use strict';

var fs         = require('fs-extra');
var ember      = require('../helpers/ember');
var existsSync = require('exists-sync');
var expect     = require('chai').expect;
var forEach    = require('lodash/collection/forEach');
var walkSync   = require('walk-sync');
var Blueprint  = require('../../lib/models/blueprint');
var path       = require('path');
var tmp        = require('../helpers/tmp');
var root       = process.cwd();
var util       = require('util');
var conf       = require('../helpers/conf');
var EOL        = require('os').EOL;

describe('Acceptance: ember new', function() {
  before(conf.setup);

  after(conf.restore);

  beforeEach(function() {
    return tmp.setup('./tmp')
      .then(function() {
        process.chdir('./tmp');
      });
  });

  afterEach(function() {
    this.timeout(10000);

    return tmp.teardown('./tmp');
  });

  function confirmBlueprintedForDir(dir) {
    return function() {
      var blueprintPath = path.join(root, dir, 'files');
      var expected      = walkSync(blueprintPath);
      var actual        = walkSync('.').sort();
      var directory     = path.basename(process.cwd());

      forEach(Blueprint.renamedFiles, function(destFile, srcFile) {
        expected[expected.indexOf(srcFile)] = destFile;
      });

      expected.sort();

      expect(directory).to.equal('foo');
      expect(expected).to.deep.equal(actual, EOL + ' expected: ' +  util.inspect(expected) +
                                             EOL + ' but got: ' +  util.inspect(actual));

    };
  }

  function confirmBlueprinted() {
    return confirmBlueprintedForDir('blueprints/app');
  }

  it('ember new foo, where foo does not yet exist, works', function() {
    return ember([
      'new',
      'foo',
      '--skip-npm',
      '--skip-bower'
    ]).then(confirmBlueprinted);
  });

  it('ember new with empty app name doesnt throw exception', function() {
    return ember([
      'new',
      ''
    ]);
  });

  it('ember new without app name doesnt throw exception', function() {
    return ember([
      'new'
    ]);
  });

  it('ember new with app name creates new directory and has a dasherized package name', function() {
    return ember([
      'new',
      'FooApp',
      '--skip-npm',
      '--skip-bower',
      '--skip-git'
    ]).then(function() {
      expect(!existsSync('FooApp'));

      var pkgJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkgJson.name).to.equal('foo-app');
    });
  });

  it('Cannot run ember new, inside of ember-cli project', function() {
    return ember([
      'new',
      'foo',
      '--skip-npm',
      '--skip-bower',
      '--skip-git'
    ]).then(function() {
      return ember([
        'new',
        'foo',
        '--skip-npm',
        '--skip-bower',
        '--skip-git'
      ]).then(function() {
        expect(!existsSync('foo'));
      });
    }).then(confirmBlueprinted);
  });

  it('ember new with blueprint uses the specified blueprint directory with a relative path', function() {
    return tmp.setup('./tmp/my_blueprint')
      .then(function() {
        return tmp.setup('./tmp/my_blueprint/files');
      })
      .then(function() {
        fs.writeFileSync('./tmp/my_blueprint/files/gitignore');
        process.chdir('./tmp');

        return ember([
          'new',
          'foo',
          '--skip-npm',
          '--skip-bower',
          '--skip-git',
          '--blueprint=./my_blueprint'
        ]);
      })
      .then(confirmBlueprintedForDir('tmp/my_blueprint'));
  });

  it('ember new with blueprint uses the specified blueprint directory with an absolute path', function() {
    return tmp.setup('./tmp/my_blueprint')
      .then(function() {
        return tmp.setup('./tmp/my_blueprint/files');
      })
      .then(function() {
        fs.writeFileSync('./tmp/my_blueprint/files/gitignore');
        process.chdir('./tmp');

        return ember([
          'new',
          'foo',
          '--skip-npm',
          '--skip-bower',
          '--skip-git',
          '--blueprint=' + path.resolve(process.cwd(), './my_blueprint')
        ]);
      })
      .then(confirmBlueprintedForDir('tmp/my_blueprint'));
  });


  it('ember new with git blueprint uses checks out the blueprint and uses it', function(){
    this.timeout(20000); // relies on GH network stuff

    return ember([
      'new',
      'foo',
      '--skip-npm',
      '--skip-bower',
      '--skip-git',
      '--blueprint=https://github.com/trek/app-blueprint-test.git'
    ]).then(function() {
      expect(existsSync('.ember-cli'));
    });
  });

  it('ember new without skip-git flag creates .git dir', function(){
    return ember([
      'new',
      'foo',
      '--skip-npm',
      '--skip-bower'
    ]).then(function() {
      expect(existsSync('.git'));
    });
  });

  it('ember new with --dry-run does not create new directory', function(){
    return ember([
      'new',
      'foo',
      '--dry-run'
    ]).then(function(){
      var cwd = process.cwd();
      expect(cwd).to.not.match(/foo/, 'does not change cwd to foo in a dry run');
      expect(!existsSync(path.join(cwd, 'foo')), 'does not create new directory');
      expect(!existsSync(path.join(cwd, '.git')), 'does not create git in current directory');
    });
  });

  it('ember new with --directory uses given directory name and has correct package name', function() {
    return ember([
      'new',
      'foo',
      '--skip-npm',
      '--skip-bower',
      '--skip-git',
      '--directory=bar'
    ]).then(function() {
      var cwd = process.cwd();
      expect(cwd).to.not.match(/foo/, 'does not use app name for directory name');
      expect(!existsSync(path.join(cwd, 'foo')), 'does not create new directory with app name');

      expect(cwd).to.match(/bar/, 'uses given directory name');
      expect(existsSync(path.join(cwd, 'bar')), 'creates new directory with specified name');

      var pkgJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkgJson.name).to.equal('foo', 'uses app name for package name');
    });
  });
});
