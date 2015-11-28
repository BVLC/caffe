/*jshint multistr: true */

'use strict';

var fs                = require('fs-extra');
var Task              = require('../../../lib/models/task');
var MockProject       = require('../../helpers/mock-project');
var MockUI            = require('../../helpers/mock-ui');
var processHelpString = require('../../helpers/process-help-string');
var expect            = require('chai').expect;
var path              = require('path');
var glob              = require('glob');
var walkSync          = require('walk-sync');
var Promise           = require('../../../lib/ext/promise');
var remove            = Promise.denodeify(fs.remove);
var EOL               = require('os').EOL;
var root              = process.cwd();
var tmp               = require('tmp-sync');
var tmproot           = path.join(root, 'tmp');
var SilentError       = require('silent-error');
var stub              = require('../../helpers/stub').stub;
var proxyquire        = require('proxyquire');
var existsSync        = require('exists-sync');
var MarkdownColor     = require('../../../lib/utilities/markdown-color');
var assign            = require('lodash/object/assign');

var existsSyncStub;
var readdirSyncStub;
var readFileSyncStub;
var renderFileStub;
var Blueprint = proxyquire('../../../lib/models/blueprint', {
  'exists-sync': function() {
    return existsSyncStub.apply(this, arguments);
  },
  'fs-extra': {
    readdirSync: function() {
      return readdirSyncStub.apply(this, arguments);
    },
    readFileSync: function() {
      return readFileSyncStub.apply(this, arguments);
    }
  },
  '../utilities/markdown-color': function() {
    return {
      renderFile: function() {
        return renderFileStub.apply(this, arguments);
      }
    };
  }
});

var defaultBlueprints = path.resolve(__dirname, '..', '..', '..', 'blueprints');
var fixtureBlueprints = path.resolve(__dirname, '..', '..', 'fixtures', 'blueprints');
var basicBlueprint    = path.join(fixtureBlueprints, 'basic');
var basicNewBlueprint = path.join(fixtureBlueprints, 'basic_2');

var defaultIgnoredFiles = Blueprint.ignoredFiles;

var basicBlueprintFiles = [
  '.ember-cli',
  '.gitignore',
  'bar',
  'foo.txt',
  'test.txt'
];

describe('Blueprint', function() {
  beforeEach(function() {
    Blueprint.ignoredFiles = defaultIgnoredFiles;

    existsSyncStub = existsSync;
    readdirSyncStub = fs.readdirSync;
    readFileSyncStub = fs.readFileSync;
    renderFileStub = MarkdownColor.prototype.renderFile;
  });

  describe('.mapFile', function() {
    it('replaces all occurences of __name__ with module name',function(){
      var path = Blueprint.prototype.mapFile('__name__/__name__-controller.js',{dasherizedModuleName: 'my-blueprint'});
      expect(path).to.equal('my-blueprint/my-blueprint-controller.js');

      path = Blueprint.prototype.mapFile('__name__/controller.js',{dasherizedModuleName: 'my-blueprint'});
      expect(path).to.equal('my-blueprint/controller.js');

      path = Blueprint.prototype.mapFile('__name__/__name__.js',{dasherizedModuleName: 'my-blueprint'});
      expect(path).to.equal('my-blueprint/my-blueprint.js');
    });
    it('accepts locals.fileMap with multiple mappings',function(){
      var locals = {};
      locals.fileMap= {
        __name__: 'user',
        __type__: 'controller',
        __path__: 'pods/users',
        __plural__: ''
      };

      var path = Blueprint.prototype.mapFile('__name__/__type____plural__.js',locals);
      expect(path).to.equal('user/controller.js');

      path = Blueprint.prototype.mapFile('__path__/__name__/__type__.js',locals);
      expect(path).to.equal('pods/users/user/controller.js');
    });
  });
  describe('.fileMapTokens', function() {
    it('adds additional tokens from fileMapTokens hook', function() {
      var blueprint = Blueprint.lookup(basicBlueprint);
      blueprint.fileMapTokens = function() {
        return {
          __foo__: function(){
            return 'foo';
          }
        };
      };
      var tokens = blueprint._fileMapTokens();
      expect(tokens.__foo__()).to.equal('foo');
    });
  });
  describe('.generateFileMap', function() {
    it('should not have locals in the fileMap', function() {
      var blueprint = Blueprint.lookup(basicBlueprint);

      var fileMapVariables = {
        pod: true,
        podPath: 'pods',
        isAddon: false,
        blueprintName: 'test',
        dasherizedModuleName: 'foo-baz',
        locals: { SOME_LOCAL_ARG: 'ARGH' }
      };

      var fileMap = blueprint.generateFileMap(fileMapVariables);
      var expected = {
        __name__: 'foo-baz',
        __path__: 'tests',
        __root__: 'app',
        __test__: 'foo-baz-test'
      };

      expect(fileMap).to.deep.equal(expected);
    });
  });
  describe('.lookup', function() {
    it('uses an explicit path if one is given', function() {
      var expectedClass = require(basicBlueprint);
      var blueprint = Blueprint.lookup(basicBlueprint);

      expect(blueprint.name).to.equal('basic');
      expect(blueprint.path).to.equal(basicBlueprint);
      expect(blueprint instanceof expectedClass).to.equal(true);
    });

    it('finds blueprints within given lookup paths', function() {
      var expectedClass = require(basicBlueprint);
      var blueprint = Blueprint.lookup('basic', {
        paths: [fixtureBlueprints]
      });

      expect(blueprint.name).to.equal('basic');
      expect(blueprint.path).to.equal(basicBlueprint);
      expect(blueprint instanceof expectedClass).to.equal(true);
    });

    it('finds blueprints in the ember-cli package', function() {
      var expectedPath = path.resolve(defaultBlueprints, 'app');
      var expectedClass = Blueprint;

      var blueprint = Blueprint.lookup('app');

      expect(blueprint.name).to.equal('app');
      expect(blueprint.path).to.equal(expectedPath);
      expect(blueprint instanceof expectedClass).to.equal(true);
    });

    it('can instantiate a blueprint that exports an object instead of a constructor', function() {
      var blueprint = Blueprint.lookup('exporting-object', {
        paths: [fixtureBlueprints]
      });

      expect(blueprint.woot).to.equal('someValueHere');
      expect(blueprint instanceof Blueprint).to.equal(true);
    });

    it('throws an error if no blueprint is found', function() {
      expect(function() {
        Blueprint.lookup('foo');
      }).to.throw('Unknown blueprint: foo');
    });

    it('returns undefined if no blueprint is found and ignoredMissing is passed', function() {
      var blueprint = Blueprint.lookup('foo', {
        ignoreMissing: true
      });

      expect(blueprint).to.equal(undefined);
    });
  });

  describe('.list', function() {
    beforeEach(function() {
      existsSyncStub = function(path) {
        return path.indexOf('package.json') === -1;
      };

      stub(Blueprint, 'defaultLookupPaths', []);
      stub(Blueprint, 'load', function(blueprintPath) {
        return {
          name: path.basename(blueprintPath)
        };
      }, true);
    });

    afterEach(function() {
      if (Blueprint.defaultLookupPaths.restore) {
        Blueprint.defaultLookupPaths.restore();
      }
      if (Blueprint.load.restore) {
        Blueprint.load.restore();
      }
    });

    it('returns a list of blueprints grouped by lookup path', function() {
      readdirSyncStub = function() {
        return ['test1', 'test2'];
      };

      var list = Blueprint.list({ paths: ['test0/blueprints'] });

      expect(list[0]).to.deep.equal({
        source: 'test0',
        blueprints: [
          {
            name: 'test1',
            overridden: false
          },
          {
            name: 'test2',
            overridden: false
          }
        ]
      });
    });

    it('overrides a blueprint of the same name from another package', function() {
      readdirSyncStub = function() {
        return ['test2'];
      };

      var list = Blueprint.list({
        paths: [
          'test0/blueprints',
          'test1/blueprints'
        ]
      });

      expect(list[0]).to.deep.equal({
        source: 'test0',
        blueprints: [
          {
            name: 'test2',
            overridden: false
          }
        ]
      });
      expect(list[1]).to.deep.equal({
        source: 'test1',
        blueprints: [
          {
            name: 'test2',
            overridden: true
          }
        ]
      });
    });
  });

  describe('help', function() {
    it('handles overridden', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      assign(blueprint, {
        overridden: true
      });

      var output = blueprint.printBasicHelp();

      var testString = processHelpString('\
      \u001b[90m(overridden) my-blueprint\u001b[39m');

      expect(output).to.equal(testString);
    });

    it('handles all possible options', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      var availableOptions = [
        {
          name: 'test-option',
          values: ['x', 'y'],
          default: 'my-def-val',
          required: true,
          aliases: ['a', { b: 'c' }],
          description: 'option desc'
        },
        {
          name: 'test-type',
          type: Boolean,
          aliases: ['a']
        }
      ];

      assign(blueprint, {
        description: 'a paragraph',
        availableOptions: availableOptions,
        anonymousOptions: ['anon-test'],
        printDetailedHelp: function() {
          expect(arguments[0]).to.equal(availableOptions);
          return 'some details';
        },
        dontShowThis: 'test'
      });

      var output = blueprint.printBasicHelp(true);

      var testString = processHelpString('\
      my-blueprint \u001b[33m<anon-test>\u001b[39m \u001b[36m<options...>\u001b[39m' + EOL + '\
        \u001b[90ma paragraph\u001b[39m' + EOL + '\
        \u001b[36m--test-option\u001b[39m\u001b[36m=x|y\u001b[39m \u001b[36m(Default: my-def-val)\u001b[39m \u001b[36m(Required)\u001b[39m' + EOL + '\
          \u001b[90maliases: -a <value>, -b (--test-option=c)\u001b[39m option desc' + EOL + '\
        \u001b[36m--test-type\u001b[39m' + EOL + '\
          \u001b[90maliases: -a\u001b[39m' + EOL + '\
some details');

      expect(output).to.equal(testString);
    });

    it('handles all possible options json', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      var availableOptions = [
        {
          type: 'my-string-type',
          showAnything: true
        },
        {
          type: function myFunctionType() {}
        }
      ];

      assign(blueprint, {
        description: 'a paragraph',
        overridden: false,
        availableOptions: availableOptions,
        anonymousOptions: ['anon-test'],
        printDetailedHelp: function() {
          expect(arguments[0]).to.equal(availableOptions);
          return 'some details';
        },
        dontShowThis: true
      });

      var json = blueprint.getJson(true);

      expect(json).to.deep.equal({
        name: 'my-blueprint',
        description: 'a paragraph',
        overridden: false,
        availableOptions: [
          {
            type: 'my-string-type',
            showAnything: true
          },
          {
            type: 'myFunctionType'
          }
        ],
        anonymousOptions: ['anon-test'],
        detailedHelp: 'some details'
      });
    });

    it('handles the simplest blueprint, to test else skipping', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      var output = blueprint.printBasicHelp();

      var testString = processHelpString('\
      my-blueprint \u001b[33m<name>\u001b[39m');

      expect(output).to.equal(testString);
    });

    it('handles the simplest option, to test else skipping', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      assign(blueprint, {
        availableOptions: [{}]
      });

      var output = blueprint.printBasicHelp();

      var testString = processHelpString('\
      my-blueprint \u001b[33m<name>\u001b[39m \u001b[36m<options...>\u001b[39m' + EOL + '\
        \u001b[36m--undefined\u001b[39m');

      expect(output).to.equal(testString);
    });

    it('don\'t print prefix if option aliases is empty', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      assign(blueprint, {
        availableOptions: [
          {
            aliases: []
          }
        ]
      });

      var output = blueprint.printBasicHelp();

      var testString = processHelpString('\
      my-blueprint \u001b[33m<name>\u001b[39m \u001b[36m<options...>\u001b[39m' + EOL + '\
        \u001b[36m--undefined\u001b[39m');

      expect(output).to.equal(testString);
    });

    it('if blueprint nulls printDetailedHelp, don\'t call it, we should deprecate this', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      assign(blueprint, {
        printDetailedHelp: null
      });

      var output = blueprint.printBasicHelp(true);

      var testString = processHelpString('\
      my-blueprint \u001b[33m<name>\u001b[39m');

      expect(output).to.equal(testString);
    });

    it('if blueprint nulls printDetailedHelp, don\'t call it json, we should deprecate this', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      assign(blueprint, {
        printDetailedHelp: null
      });

      var json = blueprint.getJson(true);

      expect(json).to.deep.equal({
        name: 'my-blueprint',
        availableOptions: [],
        anonymousOptions: ['name']
      });
    });

    it('if printDetailedHelp returns falsy, don\'t attach property detailedHelp', function() {
      var blueprint = new Blueprint('path/to/my-blueprint');

      stub(blueprint, 'printDetailedHelp', '');

      var json = blueprint.getJson(true);

      expect(blueprint.printDetailedHelp.called).to.equal(1);
      expect(json).to.not.have.property('detailedHelp');
    });

    it('handles extra help', function() {
      existsSyncStub = function() {
        return true;
      };
      renderFileStub = function() {
        expect(arguments[1].indent).to.equal('        ');
        return 'test-file';
      };

      var blueprint = new Blueprint('path/to/my-blueprint');

      var help = blueprint.printDetailedHelp();

      expect(help).to.equal('test-file');
    });

    it('handles no extra help', function() {
      existsSyncStub = function() {
        return false;
      };
      renderFileStub = function() {
        expect.fail(0, 1, 'should not call MarkdownColor.renderFile');
      };

      var blueprint = new Blueprint('path/to/my-blueprint');

      var help = blueprint.printDetailedHelp();

      expect(help).to.equal('');
    });
  });

  it('exists', function() {
    var blueprint = new Blueprint(basicBlueprint);
    expect(!!blueprint).to.equal(true);
  });

  it('derives name from path', function() {
    var blueprint = new Blueprint(basicBlueprint);
    expect(blueprint.name).to.equal('basic');
  });

  describe('filesPath', function() {
    it('returns the blueprints default files path', function() {
      var blueprint = new Blueprint(basicBlueprint);

      expect(blueprint.filesPath()).to.equal(path.join(basicBlueprint, 'files'));
    });
  });

  describe('basic blueprint installation', function() {
    var BasicBlueprintClass = require(basicBlueprint);
    var blueprint;
    var ui;
    var project;
    var options;
    var tmpdir;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new BasicBlueprintClass(basicBlueprint);
      ui        = new MockUI();
      project   = new MockProject();
      options   = {
        ui: ui,
        project: project,
        target: tmpdir
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('installs basic files', function() {
      expect(!!blueprint).to.equal(true);

      return blueprint.install(options)
        .then(function() {
          var actualFiles = walkSync(tmpdir).sort();
          var output = ui.output.trim().split(EOL);

          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/create.* .ember-cli/);
          expect(output.shift()).to.match(/create.* .gitignore/);
          expect(output.shift()).to.match(/create.* bar/);
          expect(output.shift()).to.match(/create.* foo.txt/);
          expect(output.shift()).to.match(/create.* test.txt/);
          expect(output.length).to.equal(0);

          expect(actualFiles).to.deep.equal(basicBlueprintFiles);

          expect( function(){
            fs.readFile(path.join(tmpdir , 'test.txt'), 'utf-8',
                function(err, content){
                    if(err){
                        throw 'error';
                    }
                    expect(content).to.match(/I AM TESTY/);
                });
            }
          ).not.to.throw();

        });
    });

    it('re-installing identical files', function() {
      return blueprint.install(options)
        .then(function() {
          var output = ui.output.trim().split(EOL);
          ui.output = '';

          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/create.* .ember-cli/);
          expect(output.shift()).to.match(/create.* .gitignore/);
          expect(output.shift()).to.match(/create.* bar/);
          expect(output.shift()).to.match(/create.* foo.txt/);
          expect(output.shift()).to.match(/create.* test.txt/);
          expect(output.length).to.equal(0);

          return blueprint.install(options);
        })
        .then(function() {
          var actualFiles = walkSync(tmpdir).sort();
          var output = ui.output.trim().split(EOL);

          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/identical.* .ember-cli/);
          expect(output.shift()).to.match(/identical.* .gitignore/);
          expect(output.shift()).to.match(/identical.* bar/);
          expect(output.shift()).to.match(/identical.* foo.txt/);
          expect(output.shift()).to.match(/identical.* test.txt/);
          expect(output.length).to.equal(0);

          expect(actualFiles).to.deep.equal(basicBlueprintFiles);
        });
    });

    it('re-installing conflicting files', function() {
      return blueprint.install(options)
        .then(function() {
          var output = ui.output.trim().split(EOL);
          ui.output = '';

          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/create.* .ember-cli/);
          expect(output.shift()).to.match(/create.* .gitignore/);
          expect(output.shift()).to.match(/create.* bar/);
          expect(output.shift()).to.match(/create.* foo.txt/);
          expect(output.shift()).to.match(/create.* test.txt/);
          expect(output.length).to.equal(0);
          var blueprintNew = new Blueprint(basicNewBlueprint);

          ui.waitForPrompt().then(function(){
            ui.inputStream.write('n' + EOL);
            return ui.waitForPrompt();
          }).then(function(){
            ui.inputStream.write('y' + EOL);
          });

          return blueprintNew.install(options);
        })
        .then(function() {
          var actualFiles = walkSync(tmpdir).sort();
          // Prompts contain \n EOL
          // Split output on \n since it will have the same affect as spliting on OS specific EOL
          var output = ui.output.trim().split('\n');
          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/Overwrite.*foo.*\?/); // Prompt
          expect(output.shift()).to.match(/Overwrite.*foo.*No, skip/);
          expect(output.shift()).to.match(/Overwrite.*test.*\?/); // Prompt
          expect(output.shift()).to.match(/Overwrite.*test.*Yes, overwrite/);
          expect(output.shift()).to.match(/identical.* \.ember-cli/);
          expect(output.shift()).to.match(/identical.* \.gitignore/);
          expect(output.shift()).to.match(/skip.* foo.txt/);
          expect(output.shift()).to.match(/overwrite.* test.txt/);
          expect(output.length).to.equal(0);

          expect(actualFiles).to.deep.equal(basicBlueprintFiles);
        });
    });

    it('installs path globPattern file', function() {
      options.targetFiles = ['foo.txt'];
      return blueprint.install(options)
        .then(function() {
          var actualFiles = walkSync(tmpdir).sort();
          var globFiles = glob.sync(path.join('**', 'foo.txt'), {
              cwd: tmpdir,
              dot: true,
              mark: true,
              strict: true
            }).sort();
          var output = ui.output.trim().split(EOL);

          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/create.* foo.txt/);
          expect(output.length).to.equal(0);

          expect(actualFiles).to.deep.equal(globFiles);
        });
    });

    it('installs multiple globPattern files', function() {
      options.targetFiles = ['foo.txt','test.txt'];
      return blueprint.install(options)
        .then(function() {
          var actualFiles = walkSync(tmpdir).sort();
          var globFiles = glob.sync(path.join('**', '*.txt'), {
              cwd: tmpdir,
              dot: true,
              mark: true,
              strict: true
            }).sort();
          var output = ui.output.trim().split(EOL);

          expect(output.shift()).to.match(/^installing/);
          expect(output.shift()).to.match(/create.* foo.txt/);
          expect(output.shift()).to.match(/create.* test.txt/);
          expect(output.length).to.equal(0);

          expect(actualFiles).to.deep.equal(globFiles);
        });
    });

    describe('called on an existing project', function() {
      beforeEach(function() {
        Blueprint.ignoredUpdateFiles.push('foo.txt');
      });

      it('ignores files in ignoredUpdateFiles', function() {
        return blueprint.install(options)
          .then(function() {
            var output = ui.output.trim().split(EOL);
            ui.output = '';

            expect(output.shift()).to.match(/^installing/);
            expect(output.shift()).to.match(/create.* .ember-cli/);
            expect(output.shift()).to.match(/create.* .gitignore/);
            expect(output.shift()).to.match(/create.* bar/);
            expect(output.shift()).to.match(/create.* foo.txt/);
            expect(output.shift()).to.match(/create.* test.txt/);
            expect(output.length).to.equal(0);

            var blueprintNew = new Blueprint(basicNewBlueprint);

            ui.waitForPrompt().then(function(){
              ui.inputStream.write('n' + EOL);
              return ui.waitForPrompt();
            }).then(function(){
              ui.inputStream.write('n' + EOL);
            });

            options.project.isEmberCLIProject = function() { return true; };

            return blueprintNew.install(options);
          })
          .then(function() {
            var actualFiles = walkSync(tmpdir).sort();
            // Prompts contain \n EOL
            // Split output on \n since it will have the same affect as spliting on OS specific EOL
            var output = ui.output.trim().split('\n');
            expect(output.shift()).to.match(/^installing/);
            expect(output.shift()).to.match(/Overwrite.*test.*\?/); // Prompt
            expect(output.shift()).to.match(/Overwrite.*test.*No, skip/);
            expect(output.shift()).to.match(/identical.* \.ember-cli/);
            expect(output.shift()).to.match(/identical.* \.gitignore/);
            expect(output.shift()).to.match(/skip.* test.txt/);
            expect(output.length).to.equal(0);

            expect(actualFiles).to.deep.equal(basicBlueprintFiles);
          });
      });
    });

    it('throws error when there is a trailing forward slash in entityName', function(){
      options.entity = { name: 'foo/' };
      expect(function() {
        blueprint.install(options);
      }).to.throw(/You specified "foo\/", but you can't use a trailing slash as an entity name with generators. Please re-run the command with "foo"./);

      options.entity = { name: 'foo\\' };
      expect(function() {
        blueprint.install(options);
      }).to.throw(/You specified "foo\\", but you can't use a trailing slash as an entity name with generators. Please re-run the command with "foo"./);

      options.entity = { name: 'foo' };
      expect(function() {
        blueprint.install(options);
      }).not.to.throw();
    });

    it('throws error when an entityName is not provided', function(){
      options.entity = { };
      expect(function() {
        blueprint.install(options);
      }).to.throw(SilentError, /The `ember generate <entity-name>` command requires an entity name to be specified./);
    });

    it('throws error when an action does not exist', function() {
      blueprint._actions = {};
      return blueprint.install(options)
        .catch(function(err) {
          expect(err.message).to.equal('Tried to call action "write" but it does not exist');
        });
    });

    it('calls normalizeEntityName hook during install', function(done){
      blueprint.normalizeEntityName = function(){ done(); };
      options.entity = { name: 'foo' };
      blueprint.install(options);
    });

    it('normalizeEntityName hook can modify the entity name', function(){
      blueprint.normalizeEntityName = function(){ return 'foo'; };
      options.entity = { name: 'bar' };

      return blueprint.install(options)
          .then(function() {
            var actualFiles = walkSync(tmpdir).sort();

            expect(actualFiles).to.deep.equal(basicBlueprintFiles);
          });
    });

    it('calls normalizeEntityName before locals hook is called', function(done) {
      blueprint.normalizeEntityName = function(){ return 'foo'; };
      blueprint.locals = function(options) {
        expect(options.entity.name).to.equal('foo');
        done();
      };
      options.entity = { name: 'bar' };
      blueprint.install(options);
    });
  });

  describe('basic blueprint uninstallation', function() {
    var BasicBlueprintClass = require(basicBlueprint);
    var blueprint;
    var ui;
    var project;
    var options;
    var tmpdir;

    function refreshUI() {
      ui = new MockUI();
      options.ui = ui;
    }

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new BasicBlueprintClass(basicBlueprint);
      project   = new MockProject();
      options   = {
        project: project,
        target: tmpdir
      };

      refreshUI();
      return blueprint.install(options)
        .then(refreshUI);
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('uninstalls basic files', function() {
      expect(!!blueprint).to.equal(true);

      return blueprint.uninstall(options)
        .then(function() {
          var actualFiles = walkSync(tmpdir);
          var output = ui.output.trim().split(EOL);

          expect(output.shift()).to.match(/^uninstalling/);
          expect(output.shift()).to.match(/remove.* .ember-cli/);
          expect(output.shift()).to.match(/remove.* .gitignore/);
          expect(output.shift()).to.match(/remove.* bar/);
          expect(output.shift()).to.match(/remove.* foo.txt/);
          expect(output.shift()).to.match(/remove.* test.txt/);
          expect(output.length).to.equal(0);

          expect(actualFiles.length).to.equal(0);

          fs.exists(path.join(tmpdir, 'test.txt'),
            function(exists) {
              expect(exists).to.be.false;
            });
        });
    });
  });

  describe('addPackageToProject', function() {
    var blueprint;
    var ui;
    var tmpdir;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('passes a packages array for addPackagesToProject', function() {
      blueprint.addPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar'}]);
      };

      blueprint.addPackageToProject('foo-bar');
    });

    it('passes a packages array with target for addPackagesToProject', function() {
      blueprint.addPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar', target: '^123.1.12'}]);
      };

      blueprint.addPackageToProject('foo-bar', '^123.1.12');
    });
  });

  describe('addPackagesToProject', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var NpmInstallTask;
    var taskNameLookedUp;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();

      blueprint.taskFor = function(name) {
        taskNameLookedUp = name;

        return new NpmInstallTask();
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('looks up the `npm-install` task', function() {
      NpmInstallTask = Task.extend({
        run: function() {}
      });

      blueprint.addPackagesToProject([{name: 'foo-bar'}]);

      expect(taskNameLookedUp).to.equal('npm-install');
    });

    it('calls the task with package names', function() {
      var packages;

      NpmInstallTask = Task.extend({
        run: function(options) {
          packages = options.packages;
        }
      });

      blueprint.addPackagesToProject([
        {name: 'foo-bar'},
        {name: 'bar-foo'}
      ]);

      expect(packages).to.deep.equal(['foo-bar', 'bar-foo']);
    });

    it('calls the task with package names and versions', function() {
      var packages;

      NpmInstallTask = Task.extend({
        run: function(options) {
          packages = options.packages;
        }
      });

      blueprint.addPackagesToProject([
        {name: 'foo-bar', target: '^123.1.12'},
        {name: 'bar-foo', target: '0.0.7'}
      ]);

      expect(packages).to.deep.equal(['foo-bar@^123.1.12', 'bar-foo@0.0.7']);
    });

    it('writes information to the ui log for a single package', function() {
      blueprint.ui = ui;

      blueprint.addPackagesToProject([
        {name: 'foo-bar', target: '^123.1.12'}
      ]);

      var output = ui.output.trim();

      expect(output).to.match(/install package.*foo-bar/);
    });

    it('writes information to the ui log for multiple packages', function() {
      blueprint.ui = ui;

      blueprint.addPackagesToProject([
        {name: 'foo-bar', target: '^123.1.12'},
        {name: 'bar-foo', target: '0.0.7'}
      ]);

      var output = ui.output.trim();

      expect(output).to.match(/install packages.*foo-bar, bar-foo/);
    });

    it('does not error if ui is not present', function() {
      delete blueprint.ui;

      blueprint.addPackagesToProject([
        {name: 'foo-bar', target: '^123.1.12'}
      ]);

      var output = ui.output.trim();

      expect(output).to.not.match(/install package.*foo-bar/);
    });

    it('runs task with --save-dev', function() {
      var saveDev;

      NpmInstallTask = Task.extend({
        run: function(options) {
          saveDev = options['save-dev'];
        }
      });

      blueprint.addPackagesToProject([
        {name: 'foo-bar', target: '^123.1.12'},
        {name: 'bar-foo', target: '0.0.7'}
      ]);

      expect(!!saveDev).to.equal(true);
    });

    it('does not use verbose mode with the task', function() {
      var verbose;

      NpmInstallTask = Task.extend({
        run: function(options) {
          verbose = options.verbose;
        }
      });

      blueprint.addPackagesToProject([
        {name: 'foo-bar', target: '^123.1.12'},
        {name: 'bar-foo', target: '0.0.7'}
      ]);

      expect(verbose).to.equal(false);
    });
  });

  describe('removePackageFromProject', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var NpmUninstallTask;
    var taskNameLookedUp;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();

      blueprint.taskFor = function(name) {
        taskNameLookedUp = name;

        return new NpmUninstallTask();
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('looks up the `npm-uninstall` task', function() {
      NpmUninstallTask = Task.extend({
        run: function() {}
      });

      blueprint.removePackageFromProject({name: 'foo-bar'});

      expect(taskNameLookedUp).to.equal('npm-uninstall');
    });

  });

  describe('removePackagesFromProject', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var NpmUninstallTask;
    var taskNameLookedUp;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();

      blueprint.taskFor = function(name) {
        taskNameLookedUp = name;

        return new NpmUninstallTask();
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('looks up the `npm-uninstall` task', function() {
      NpmUninstallTask = Task.extend({
        run: function() {}
      });

      blueprint.removePackagesFromProject([{name: 'foo-bar'}]);

      expect(taskNameLookedUp).to.equal('npm-uninstall');
    });

    it('calls the task with package names', function() {
      var packages;

      NpmUninstallTask = Task.extend({
        run: function(options) {
          packages = options.packages;
        }
      });

      blueprint.removePackagesFromProject([
        {name: 'foo-bar'},
        {name: 'bar-foo'}
      ]);

      expect(packages).to.deep.equal(['foo-bar', 'bar-foo']);
    });

    it('writes information to the ui log for a single package', function() {
      blueprint.ui = ui;

      blueprint.removePackagesFromProject([
        {name: 'foo-bar'}
      ]);

      var output = ui.output.trim();

      expect(output).to.match(/uninstall package.*foo-bar/);
    });

    it('writes information to the ui log for multiple packages', function() {
      blueprint.ui = ui;

      blueprint.removePackagesFromProject([
        {name: 'foo-bar'},
        {name: 'bar-foo'}
      ]);

      var output = ui.output.trim();

      expect(output).to.match(/uninstall packages.*foo-bar, bar-foo/);
    });

    it('does not error if ui is not present', function() {
      delete blueprint.ui;

      blueprint.removePackagesFromProject([
        {name: 'foo-bar'}
      ]);

      var output = ui.output.trim();

      expect(output).to.not.match(/uninstall package.*foo-bar/);
    });

    it('runs task with --save-dev', function() {
      var saveDev;

      NpmUninstallTask = Task.extend({
        run: function(options) {
          saveDev = options['save-dev'];
        }
      });

      blueprint.removePackagesFromProject([
        {name: 'foo-bar'},
        {name: 'bar-foo'}
      ]);

      expect(!!saveDev).to.equal(true);
    });

    it('does not use verbose mode with the task', function() {
      var verbose;

      NpmUninstallTask = Task.extend({
        run: function(options) {
          verbose = options.verbose;
        }
      });

      blueprint.removePackagesFromProject([
        {name: 'foo-bar'},
        {name: 'bar-foo'}
      ]);

      expect(verbose).to.equal(false);
    });
  });

  describe('addBowerPackageToProject', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var BowerInstallTask;
    var taskNameLookedUp;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();
      blueprint.ui = ui;
      blueprint.taskFor = function(name) {
        taskNameLookedUp = name;

        return new BowerInstallTask();
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('passes a packages array for addBowerPackagesToProject', function() {
      blueprint.addBowerPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar', source: 'foo-bar', target: '*'}]);
      };

      blueprint.addBowerPackageToProject('foo-bar');
    });

    it('passes a packages array with target for addBowerPackagesToProject', function() {
      blueprint.addBowerPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar', source: 'foo-bar', target: '1.0.0'}]);
      };

      blueprint.addBowerPackageToProject('foo-bar', '1.0.0');
    });

    it('correctly handles local package naming, with a numbered pkg version', function() {
      blueprint.addBowerPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar-local', target: '1.0.0', source: 'foo-bar'}]);
      };

      blueprint.addBowerPackageToProject('foo-bar-local', 'foo-bar#1.0.0');
    });

    it('correctly handles local package naming, with a non-versioned package', function() {
      blueprint.addBowerPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar-local', target: '*', source: 'http://twitter.github.io/bootstrap/assets/bootstrap'}]);
      };

      blueprint.addBowerPackageToProject('foo-bar-local', 'http://twitter.github.io/bootstrap/assets/bootstrap');
    });

    it('correctly handles a single versioned package descriptor as argument (1) (DEPRECATED)', function() {
      blueprint.ui = ui;
      blueprint.addBowerPackagesToProject = function(packages) {
        expect(packages).to.deep.equal([{name: 'foo-bar', target: '1.11.1', source: 'foo-bar'}]);
      };

      blueprint.addBowerPackageToProject('foo-bar#1.11.1');
    });
  });

  describe('addBowerPackagesToProject', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var BowerInstallTask;
    var taskNameLookedUp;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();

      blueprint.taskFor = function(name) {
        taskNameLookedUp = name;

        return new BowerInstallTask();
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('looks up the `bower-install` task', function() {
      BowerInstallTask = Task.extend({
        run: function() {}
      });
      blueprint.addBowerPackagesToProject([{name: 'foo-bar'}]);

      expect(taskNameLookedUp).to.equal('bower-install');
    });

    it('calls the task with the package names', function() {
      var packages;

      BowerInstallTask = Task.extend({
        run: function(options) {
          packages = options.packages;
        }
      });

      blueprint.addBowerPackagesToProject([
        {name: 'foo-bar'},
        {name: 'bar-foo'}
      ]);

      expect(packages).to.deep.equal(['foo-bar=foo-bar', 'bar-foo=bar-foo']);
    });

    it('uses the provided target (version, range, sha, etc)', function() {
      var packages;

      BowerInstallTask = Task.extend({
        run: function(options) {
          packages = options.packages;
        }
      });

      blueprint.addBowerPackagesToProject([
        {name: 'foo-bar', target: '~1.0.0'},
        {name: 'bar-foo', target: '0.7.0'}
      ]);

      expect(packages).to.deep.equal(['foo-bar=foo-bar#~1.0.0', 'bar-foo=bar-foo#0.7.0']);
    });

    it('properly parses a variety of bower package endpoints', function() {
      var packages;

      BowerInstallTask = Task.extend({
        run: function(options) {
          packages = options.packages;
        }
      });

      blueprint.addBowerPackagesToProject([
        {name: '',          source: 'jquery', target: '~2.0.0'},
        {name: 'backbone',  source: 'backbone-amd', target: '~1.0.0'},
        {name: 'bootstrap', source: 'http://twitter.github.io/bootstrap/assets/bootstrap', target: '*'}
      ]);

      expect(packages).to.deep.equal([
        // standard local name, versioned bower pkg
        'jquery#~2.0.0',
        // custom local name, versioned bower pkg
        'backbone=backbone-amd#~1.0.0',
        // no numbered version, custom local name
        'bootstrap=http://twitter.github.io/bootstrap/assets/bootstrap'
      ]);
    });

    it('uses uses verbose mode with the task', function() {
      var verbose;

      BowerInstallTask = Task.extend({
        run: function(options) {
          verbose = options.verbose;
        }
      });

      blueprint.addBowerPackagesToProject([
        {name: 'foo-bar', target: '~1.0.0'},
        {name: 'bar-foo', target: '0.7.0'}
      ]);

      expect(verbose).to.equal(true);
    });
  });

  describe('addAddonToProject', function() {
    var blueprint;
    var ui;
    var tmpdir;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('passes a packages array for addAddonsToProject', function() {
      blueprint.addAddonsToProject = function(options) {
        expect(options.packages).to.deep.equal(['foo-bar']);
      };

      blueprint.addAddonToProject('foo-bar');
    });

    it('passes a packages array with target for addAddonsToProject', function() {
      blueprint.addAddonsToProject = function(options) {
        expect(options.packages).to.deep.equal([{name: 'foo-bar', target: '^123.1.12'}]);
      };

      blueprint.addAddonToProject({name: 'foo-bar', target: '^123.1.12'});
    });
  });

  describe('addAddonsToProject', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var AddonInstallTask;
    var taskNameLookedUp;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();

      blueprint.taskFor = function(name) {
        taskNameLookedUp = name;

        return new AddonInstallTask();
      };
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('looks up the `addon-install` task', function() {
      AddonInstallTask = Task.extend({
        run: function() {}
      });

      blueprint.addAddonsToProject({ packages: ['foo-bar'] });

      expect(taskNameLookedUp).to.equal('addon-install');
    });

    it('calls the task with package name', function() {
      var pkg;

      AddonInstallTask = Task.extend({
        run: function(options) {
          pkg = options['packages'];
        }
      });

      blueprint.addAddonsToProject({ packages: ['foo-bar', 'baz-bat'] });

      expect(pkg).to.deep.equal(['foo-bar', 'baz-bat']);
    });

    it('calls the task with correctly parsed options', function() {
      var pkg, args, bluOpts;

      AddonInstallTask = Task.extend({
        run: function(options) {
          pkg  = options['packages'];
          args = options['extraArgs'];
          bluOpts = options['blueprintOptions'];
        }
      });

      blueprint.addAddonsToProject({
        packages: [
          {
            name: 'foo-bar',
            target: '1.0.0'
          },
          'stuff-things',
          'baz-bat@0.0.1'
        ],
        extraArgs: ['baz'],
        blueprintOptions: '-foo'
      });

      expect(pkg).to.deep.equal(['foo-bar@1.0.0', 'stuff-things', 'baz-bat@0.0.1']);
      expect(args).to.deep.equal(['baz']);
      expect(bluOpts).to.equal('-foo');
    });

    it('writes information to the ui log for a single package', function() {
      blueprint.ui = ui;

      blueprint.addAddonsToProject({
        packages: [{
          name: 'foo-bar',
          target: '^123.1.12'
        }]
      });

      var output = ui.output.trim();

      expect(output).to.match(/install addon.*foo-bar/);
    });

    it('writes information to the ui log for multiple packages', function() {
      blueprint.ui = ui;

      blueprint.addAddonsToProject({
        packages: [
          {
            name: 'foo-bar',
            target: '1.0.0'
          },
          'stuff-things',
          'baz-bat@0.0.1'
        ]
      });

      var output = ui.output.trim();

      expect(output).to.match(/install addons.*foo-bar@1.0.0,.*stuff-things,.*baz-bat@0.0.1/);
    });

    it('does not error if ui is not present', function() {
      delete blueprint.ui;

      blueprint.addAddonsToProject({
        packages: [{
          name: 'foo-bar',
          target: '^123.1.12'
        }]
      });

      var output = ui.output.trim();

      expect(output).to.not.match(/install addon.*foo-bar/);
    });
  });

  describe('load', function(){
    var blueprint;
    it('loads and returns a blueprint object', function() {
      blueprint = Blueprint.load(basicBlueprint);
      expect(blueprint).to.be.an('object');
      expect(blueprint.name).to.equal('basic');
    });
    it('loads only blueprints with an index.js', function() {
      expect(Blueprint.load(path.join(fixtureBlueprints, '.notablueprint'))).to.be.empty;
    });
  });

  describe('insertIntoFile', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var project;
    var filename;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();
      project   = new MockProject();

      // normally provided by `install`, but mocked here for testing
      project.root = tmpdir;
      blueprint.project = project;

      filename = 'foo-bar-baz.txt';
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('will create the file if not already existing', function() {
      var toInsert = 'blahzorz blammo';

      return blueprint.insertIntoFile(filename, toInsert)
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents.indexOf(toInsert) > -1).to.equal(true, 'contents were inserted');
          expect(result.originalContents).to.equal('', 'returned object should contain original contents');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was modified');
          expect(contents).to.equal(result.contents, 'returned object should contain contents');
        });
    });

    it('will insert into the file if it already exists', function() {
      var toInsert = 'blahzorz blammo';
      var originalContent = 'some original content\n';
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert)
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal(originalContent + toInsert, 'inserted contents should be appended to original');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was modified');
        });
    });

    it('will not insert into the file if it already contains the content', function() {
      var toInsert = 'blahzorz blammo';
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, toInsert, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert)
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal(toInsert, 'contents should be unchanged');
          expect(result.inserted).to.equal(false, 'inserted should indicate that the file was not modified');
        });
    });

    it('will insert into the file if it already contains the content if force option is passed', function() {
      var toInsert = 'blahzorz blammo';
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, toInsert, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, { force: true })
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal(toInsert + toInsert, 'contents should be unchanged');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was not modified');
        });
    });

    it('will insert into the file after a specified string if options.after is specified', function(){
      var toInsert = 'blahzorz blammo';
      var line1 = 'line1 is here';
      var line2 = 'line2 here';
      var line3 = 'line3';
      var originalContent = [line1, line2, line3].join(EOL);
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, {after: line2 + EOL})
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal([line1, line2, toInsert, line3].join(EOL),
                       'inserted contents should be inserted after the `after` value');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was modified');
        });
    });

    it('will insert into the file after the first instance of options.after only', function(){
      var toInsert = 'blahzorz blammo';
      var line1 = 'line1 is here';
      var line2 = 'line2 here';
      var line3 = 'line3';
      var originalContent = [line1, line2, line2, line3].join(EOL);
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, {after: line2 + EOL})
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal([line1, line2, toInsert, line2, line3].join(EOL),
                       'inserted contents should be inserted after the `after` value');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was modified');
        });
    });

    it('will insert into the file before a specified string if options.before is specified', function(){
      var toInsert = 'blahzorz blammo';
      var line1 = 'line1 is here';
      var line2 = 'line2 here';
      var line3 = 'line3';
      var originalContent = [line1, line2, line3].join(EOL);
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, {before: line2 + EOL})
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal([line1, toInsert, line2, line3].join(EOL),
                       'inserted contents should be inserted before the `before` value');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was modified');
        });
    });

    it('will insert into the file before the first instance of options.before only', function(){
      var toInsert = 'blahzorz blammo';
      var line1 = 'line1 is here';
      var line2 = 'line2 here';
      var line3 = 'line3';
      var originalContent = [line1, line2, line2, line3].join(EOL);
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, {before: line2 + EOL})
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal([line1, toInsert, line2, line2, line3].join(EOL),
                       'inserted contents should be inserted after the `after` value');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(true, 'inserted should indicate that the file was modified');
        });
    });


    it('it will make no change if options.after is not found in the original', function(){
      var toInsert = 'blahzorz blammo';
      var originalContent = 'the original content';
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, {after: 'not found' + EOL})
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal(originalContent, 'original content is unchanged');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(false, 'inserted should indicate that the file was not modified');
        });
    });

    it('it will make no change if options.before is not found in the original', function(){
      var toInsert = 'blahzorz blammo';
      var originalContent = 'the original content';
      var filePath = path.join(project.root, filename);

      fs.writeFileSync(filePath, originalContent, { encoding: 'utf8' });

      return blueprint.insertIntoFile(filename, toInsert, {before: 'not found' + EOL})
        .then(function(result) {
          var contents = fs.readFileSync(path.join(project.root, filename), { encoding: 'utf8' });

          expect(contents).to.equal(originalContent, 'original content is unchanged');
          expect(result.originalContents).to.equal(originalContent, 'returned object should contain original contents');
          expect(result.inserted).to.equal(false, 'inserted should indicate that the file was not modified');
        });
    });
  });

  describe('lookupBlueprint', function() {
    var blueprint;
    var ui;
    var tmpdir;
    var project;
    var filename;

    beforeEach(function() {
      tmpdir    = tmp.in(tmproot);
      blueprint = new Blueprint(basicBlueprint);
      ui        = new MockUI();
      project   = new MockProject();

      // normally provided by `install`, but mocked here for testing
      project.root = tmpdir;
      blueprint.project = project;
      project.blueprintLookupPaths = function() {
        return [fixtureBlueprints];
      };

      filename = 'foo-bar-baz.txt';
    });

    afterEach(function() {
      return remove(tmproot);
    });

    it('can lookup other Blueprints from the project blueprintLookupPaths', function() {
      var result = blueprint.lookupBlueprint('basic_2');

      expect(result.description).to.equal('Another basic blueprint');
    });

    it('can find internal blueprints', function() {
      var result = blueprint.lookupBlueprint('controller');

      expect(result.description).to.equal('Generates a controller.');
    });
  });

  describe('._generateFileMapVariables', function() {
    var blueprint;
    var project;
    var moduleName;
    var locals;
    var options;
    var result;
    var expectation;

    beforeEach(function() {
      blueprint = new Blueprint(basicBlueprint);
      project = new MockProject();
      moduleName = project.name();
      locals = {};

      blueprint.project = project;

      options = {
        project: project
      };

      expectation = {
        blueprintName: 'basic',
        dasherizedModuleName: 'mock-project',
        hasPathToken: undefined,
        inAddon: false,
        inDummy: false,
        inRepoAddon: undefined,
        locals: {},
        originBlueprintName: 'basic',
        pod: undefined,
        podPath: ''
      };
    });

    it('should create the correct default fileMapVariables', function() {
      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should use the moduleName method argument for moduleName', function() {
      moduleName = 'foo';
      expectation.dasherizedModuleName = 'foo';

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should use the locals method argument for its locals value', function() {
      locals = { foo: 'bar' };
      expectation.locals = locals;

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should use the option.originBlueprintName value as its originBlueprintName if included in the options hash', function() {
      options.originBlueprintName = 'foo';
      expectation.originBlueprintName = 'foo';

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should include a podPath if the project\'s podModulePrefix is defined', function() {
      blueprint.project.config = function() {
        return {
          podModulePrefix: 'foo/bar'
        };
      };

      expectation.podPath = 'bar';

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should include an inAddon and inDummy flag of true if the project is an addon', function () {
      options.dummy = true;

      blueprint.project.isEmberCLIAddon = function() {
        return true;
      };

      expectation.inAddon = true;
      expectation.inDummy = true;

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should include an inAddon and inRepoAddon flag of true if options.inRepoAddon is true', function() {
      options.inRepoAddon = true;

      expectation.inRepoAddon = true;
      expectation.inAddon = true;

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });

    it('should have a hasPathToken flag of true if the blueprint hasPathToken is true', function() {
      blueprint.hasPathToken = true;

      expectation.hasPathToken = true;

      result = blueprint._generateFileMapVariables(moduleName, locals, options);

      expect(result).to.eql(expectation);
    });
  });

  describe('._locals', function() {
    var blueprint;
    var project;
    var options;
    var result;
    var expectation;

    beforeEach(function() {
      blueprint = new Blueprint(basicBlueprint);
      project = new MockProject();

      blueprint._generateFileMapVariables = function() {
        return {};
      };

      blueprint.generateFileMap = function() {
        return {};
      };

      options = {
        project: project
      };

      expectation = {
        'camelizedModuleName': 'mockProject',
        'classifiedModuleName': 'MockProject',
        'classifiedPackageName': 'MockProject',
        'dasherizedModuleName': 'mock-project',
        'dasherizedPackageName': 'mock-project',
        'decamelizedModuleName': 'mock-project',
        'fileMap': {}
      };
    });

    it('should return a default object if no custom options are passed', function() {
      result = blueprint._locals(options);

      expect(result).to.eql(expectation);
    });

    it('it should call the locals method with the correct arguments', function() {
      blueprint.locals = function (opts) {
        expect(opts).to.equal(options);
      };

      blueprint._locals(options);
    });

    it('should call _generateFileMapVariables with the correct arguments', function() {
      blueprint.locals = function() {
        return { foo: 'bar' };
      };

      blueprint._generateFileMapVariables = function(modName, lcls, opts) {
        expect(modName).to.equal('mock-project');
        expect(lcls).to.eql({ foo: 'bar' });
        expect(opts).to.eql(opts);
      };

      blueprint._locals(options);
    });

    it('should call generateFileMap with the correct arguments', function() {
      blueprint._generateFileMapVariables = function() {
        return { bar: 'baz' };
      };

      blueprint.generateFileMap = function(fileMapVariables) {
        expect(fileMapVariables).to.eql({ bar: 'baz' });
      };

      blueprint._locals(options);
    });

    it('should use the options.entity.name as its moduleName if its value is defined', function() {
      options.entity = {
        name: 'foo'
      };

      expectation.camelizedModuleName = 'foo';
      expectation.classifiedModuleName = 'Foo';
      expectation.dasherizedModuleName = 'foo';
      expectation.decamelizedModuleName = 'foo';

      result = blueprint._locals(options);

      expect(result).to.eql(expectation);
    });

    it('should update its fileMap values to match the generateFileMap result', function() {
      blueprint.generateFileMap = function() {
        return { foo: 'bar' };
      };

      expectation.fileMap = { foo: 'bar' };

      result = blueprint._locals(options);

      expect(result).to.eql(expectation);
    });

    it('should return an object containing custom local values', function() {
      blueprint.locals = function() {
        return { foo: 'bar' };
      };

      expectation.foo = 'bar';

      result = blueprint._locals(options);

      expect(result).to.eql(expectation);
    });
  });
});
