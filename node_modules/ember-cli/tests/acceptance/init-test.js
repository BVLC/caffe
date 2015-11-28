'use strict';

var ember      = require('../helpers/ember');
var expect     = require('chai').expect;
var walkSync   = require('walk-sync');
var existsSync = require('exists-sync');
var glob       = require('glob');
var Blueprint  = require('../../lib/models/blueprint');
var path       = require('path');
var tmp        = require('../helpers/tmp');
var root       = process.cwd();
var util       = require('util');
var conf       = require('../helpers/conf');
var minimatch  = require('minimatch');
var intersect  = require('lodash/array/intersection');
var remove     = require('lodash/array/remove');
var forEach    = require('lodash/collection/forEach');
var any        = require('lodash/collection/some');
var EOL        = require('os').EOL;

var defaultIgnoredFiles = Blueprint.ignoredFiles;

describe('Acceptance: ember init', function() {
  this.timeout(20000);

  before(function() {
    conf.setup();
  });

  after(function() {
    conf.restore();
  });

  beforeEach(function() {
    Blueprint.ignoredFiles = defaultIgnoredFiles;

    return tmp.setup('./tmp')
      .then(function() {
        process.chdir('./tmp');
      });
  });

  afterEach(function() {
    return tmp.teardown('./tmp');
  });

  function confirmBlueprinted() {
    var blueprintPath = path.join(root, 'blueprints', 'app', 'files');
    var expected      = walkSync(blueprintPath).sort();
    var actual        = walkSync('.').sort();

    forEach(Blueprint.renamedFiles, function(destFile, srcFile) {
      expected[expected.indexOf(srcFile)] = destFile;
    });

    removeIgnored(expected);
    removeIgnored(actual);

    expected.sort();

    expect(expected).to.deep.equal(actual, EOL + ' expected: ' +  util.inspect(expected) +
                                           EOL + ' but got: ' +  util.inspect(actual));
  }

  function confirmGlobBlueprinted(pattern) {
    var blueprintPath = path.join(root, 'blueprints', 'app', 'files');
    var actual        = pickSync('.', pattern);
    var expected      = intersect(actual, pickSync(blueprintPath, pattern));

    removeIgnored(expected);
    removeIgnored(actual);

    expected.sort();

    expect(expected).to.deep.equal(actual, EOL + ' expected: ' +  util.inspect(expected) +
                                           EOL + ' but got: ' +  util.inspect(actual));
  }

  function pickSync(filePath, pattern) {
    return glob.sync(path.join('**', pattern), {
        cwd: filePath,
        dot: true,
        mark: true,
        strict: true
      }).sort();
  }
  function removeIgnored(array) {
    remove(array, function(fn) {
      return any(Blueprint.ignoredFiles, function(ignoredFile) {
        return minimatch(fn, ignoredFile, {
          matchBase: true
        });
      });
    });
  }

  it('ember init', function() {
    return ember([
      'init',
      '--skip-npm',
      '--skip-bower',
    ]).then(confirmBlueprinted);
  });

  it('ember init can run in created folder', function() {
    return tmp.setup('./tmp/foo')
      .then(function() {
        process.chdir('./tmp/foo');
      })
      .then(function() {
        return ember([
          'init',
          '--skip-npm',
          '--skip-bower'
        ]);
      })
      .then(confirmBlueprinted)
      .then(function() {
        return tmp.teardown('./tmp/foo');
      });
  });

  it('init an already init\'d folder', function() {
    return ember([
      'init',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() {
      return ember([
        'init',
        '--skip-npm',
        '--skip-bower'
      ]);
    })
    .then(confirmBlueprinted);
  });

  it('init a single file', function() {
    return ember([
      'init',
      'app.js',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() { return 'app.js'; })
    .then(confirmGlobBlueprinted);
  });

  it('init a single file on already init\'d folder', function() {
    return ember([
      'init',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() {
      return ember([
        'init',
        'app.js',
        '--skip-npm',
        '--skip-bower'
      ]);
    })
    .then(confirmBlueprinted);
  });

  it('init multiple files by glob pattern', function() {
    return ember([
      'init',
      'app/**',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() { return 'app/**'; })
    .then(confirmGlobBlueprinted);
  });

  it('init multiple files by glob pattern on already init\'d folder', function() {
    return ember([
      'init',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() {
      return ember([
        'init',
        'app/**',
        '--skip-npm',
        '--skip-bower'
      ]);
    })
    .then(confirmBlueprinted);
  });

  it('init multiple files by glob patterns', function() {
    return ember([
      'init',
      'app/**',
      '{package,bower}.json',
      'resolver.js',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() { return '{app/**,{package,bower}.json,resolver.js}'; })
    .then(confirmGlobBlueprinted);
  });

  it('init multiple files by glob patterns on already init\'d folder', function() {
    return ember([
      'init',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() {
      return ember([
        'init',
        'app/**',
        '{package,bower}.json',
        'resolver.js',
        '--skip-npm',
        '--skip-bower'
      ]);
    })
    .then(confirmBlueprinted);
  });

  it('should not create .git folder', function() {
    return ember([
      'init',
      '--skip-npm',
      '--skip-bower'
    ])
    .then(function() {
      expect(!existsSync('.git'));
    });
  });

});
