'use strict';
/*jshint expr: true*/

var expect         = require('chai').expect;
var commandOptions = require('../../factories/command-options');
var Command        = require('../../../lib/models/command');
var assign         = require('lodash/object/assign');
var Yam            = require('yam');

var ServeCommand = Command.extend({
  name: 'serve',
  aliases: ['server', 's'],
  availableOptions: [
    { name: 'port', type: Number, default: 4200 },
    { name: 'host', type: String, default: '0.0.0.0' },
    { name: 'proxy',  type: String },
    { name: 'live-reload',  type: Boolean, default: true, aliases: ['lr'] },
    { name: 'live-reload-port', type: Number, description: '(Defaults to port number + 31529)' },
    { name: 'environment', type: String, default: 'development' }
  ],
  run: function() {}
});

var DevelopEmberCLICommand = Command.extend({
  name: 'develop-ember-cli',
  works: 'everywhere',
  availableOptions: [
    { name: 'package-name', key: 'packageName', type: String, required: true }
  ],
  run: function() {}
});

var InsideProjectCommand = Command.extend({
  name: 'inside-project',
  works: 'insideProject',
  run: function() {}
});

var OutsideProjectCommand = Command.extend({
  name: 'outside-project',
  works: 'outsideProject',
  run: function() {}
});

var OptionsAliasCommand = Command.extend({
  name: 'options-alias',
  availableOptions: [{
    name: 'taco',
    type: String,
    default: 'traditional',
    aliases: [
      { 'hard-shell': 'hard-shell' },
      { 'soft-shell': 'soft-shell' }
    ]
  },
  {
    name: 'spicy',
    type: Boolean,
    default: true,
    aliases: [
      { 'mild': false }
    ]
  },
  {
    name: 'display-message',
    type: String,
    aliases: [
      'dm',
      { 'hw': 'Hello world' }
    ]
  }],
  run: function() {}
});

describe('models/command.js', function() {
  var ui;
  var config;
  var options;

  before(function() {
    config = new Yam('ember-cli', {
      secondary: process.cwd() + '/tests/fixtures/home',
      primary:   process.cwd() + '/tests/fixtures/project'
    });
  });

  beforeEach(function() {
    options = commandOptions();
    ui = options.ui;
  });

  it('parseArgs() should parse the command options.', function() {
    expect(new ServeCommand(options).parseArgs(['--port', '80'])).to.have.deep.property('options.port', 80);
  });

  it('parseArgs() should get command options from the config file and command line', function() {
    expect(new ServeCommand(assign(options, {
      settings: config.getAll()
    })).parseArgs(['--port', '789'])).to.deep.equal({
      options: {
        port: 789,
        environment: 'mock-development',
        host: '0.1.0.1',
        proxy: 'http://iamstef.net/ember-cli',
        liveReload: false,
        checkForUpdates: true
      },
      args: []
    });
  });

  it('parseArgs() should set default option values.', function() {
    expect(new ServeCommand(options).parseArgs([])).to.have.deep.property('options.port', 4200);
  });

  it('parseArgs() should return args too.', function() {
    expect(new ServeCommand(assign(options, {
      settings: config.getAll()
    })).parseArgs(['foo', '--port', '80'])).to.deep.equal({
      args: ['foo'],
      options: {
        environment: 'mock-development',
        host: '0.1.0.1',
        proxy: 'http://iamstef.net/ember-cli',
        liveReload: false,
        port: 80,
        checkForUpdates: true
      }
    });
  });

  it('parseArgs() should warn if an option is invalid.', function() {
    new ServeCommand(assign(options, {
      settings: config.getAll()
    })).parseArgs(['foo', '--envirmont', 'production']);
    expect(ui.output).to.match(/The option '--envirmont' is not registered with the serve command. Run `ember serve --help` for a list of supported options./);
  });

  it('parseArgs() should parse shorthand options.', function() {
    expect(new ServeCommand(options).parseArgs(['-e', 'tacotown'])).to.have.deep.property('options.environment', 'tacotown');
  });

  it('parseArgs() should parse shorthand dasherized options.', function() {
    expect(new ServeCommand(options).parseArgs(['-lr', 'false'])).to.have.deep.property('options.liveReload', false);
  });

  it('validateAndRun() should print a message if a required option is missing.', function() {
    return new DevelopEmberCLICommand(options).validateAndRun([]).then(function() {
      expect(ui.output).to.match(/requires the option.*package-name/);
    });
  });

  it('validateAndRun() should print a message if outside a project and command is not valid there.', function() {
    return new InsideProjectCommand(assign(options, {
      project: { isEmberCLIProject: function() { return false; } }
    })).validateAndRun([]).catch(function(reason) {
      expect(reason.message).to.match(/You have to be inside an ember-cli project/);
    });
  });

  it('validateAndRun() should print a message if inside a project and command is not valid there.', function() {
    return new OutsideProjectCommand(options).validateAndRun([]).catch(function(reason) {
      expect(reason.message).to.match(/You cannot use.*inside an ember-cli project/);
    });
  });

  it('availableOptions with aliases should work.', function() {
    expect(new OptionsAliasCommand(options).parseArgs(['-soft-shell'])).to.deep.equal({
      options: {
        taco: 'soft-shell',
        spicy: true
      },
      args: []
    });
  });

  it('availableOptions with aliases should work with minimum characters.', function() {
    expect(new OptionsAliasCommand(options).parseArgs(['-so'])).to.deep.equal({
      options: {
        taco: 'soft-shell',
        spicy: true
      },
      args: []
    });
  });

  it('availableOptions with aliases should work with hyphenated options', function() {
    expect(new OptionsAliasCommand(options).parseArgs(['-dm', 'hi'])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        displayMessage: 'hi'
      },
      args: []
    });

    expect(new OptionsAliasCommand(options).parseArgs(['-hw'])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        displayMessage: 'Hello world'
      },
      args: []
    });
  });

  it('registerOptions() should allow adding availableOptions.', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var extendedAvailableOptions = [{
      name: 'filling',
      type: String,
      default: 'adobada',
      aliases: [
        { 'carne-asada': 'carne-asada' },
        { 'carnitas': 'carnitas' },
        { 'fish': 'fish' }
      ]
    }];

    optionsAlias.registerOptions({ availableOptions: extendedAvailableOptions });
    // defaults
    expect(optionsAlias.parseArgs([])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'adobada'
      },
      args: []
    });
    // shorthand
    expect(optionsAlias.parseArgs(['-carne'])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'carne-asada'
      },
      args: []
    });
    // last argument wins
    expect(optionsAlias.parseArgs(['-carne','-fish'])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'fish'
      },
      args: []
    });
  });

  it('registerOptions() should allow overriding availableOptions.', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var extendedAvailableOptions = [{
      name: 'filling',
      type: String,
      default: 'adobada',
      aliases: [
        { 'carne-asada': 'carne-asada' },
        { 'carnitas': 'carnitas' },
        { 'fish': 'fish' }
      ]
    }];
    var duplicateExtendedAvailableOptions = [{
      name: 'filling',
      type: String,
      default: 'carnitas',
      aliases: [
        { 'pollo-asado': 'pollo-asado' },
        { 'carne-asada': 'carne-asada' }
      ]
    }];

    optionsAlias.registerOptions({ availableOptions: extendedAvailableOptions });
    // default
    expect(optionsAlias.parseArgs([])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'adobada'
      },
      args: []
    });
    // shorthand
    expect(optionsAlias.parseArgs(['-carne'])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'carne-asada'
      },
      args: []
    });
    optionsAlias.registerOptions({ availableOptions: duplicateExtendedAvailableOptions });
    // override default
    expect(optionsAlias.parseArgs([])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'carnitas'
      },
      args: []
    });
    // last argument wins
    expect(optionsAlias.parseArgs(['-fish', '-pollo'])).to.deep.equal({
      options: {
        taco: 'traditional',
        spicy: true,
        filling: 'pollo-asado'
      },
      args: []
    });
  });

  it('registerOptions() should not allow aliases with the same name.', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var extendedAvailableOptions = [
      {
        name: 'filling',
        type: String,
        default: 'adobada',
        aliases: [
          { 'carne-asada': 'carne-asada' },
          { 'carnitas': 'carnitas' },
          { 'fish': 'fish' }
        ]
      },
      {
        name: 'favorite',
        type: String,
        default: 'adobada',
        aliases: [
          { 'carne-asada': 'carne-asada' },
          { 'carnitas': 'carnitas' },
          { 'fish': 'fish' }
        ]
      }
    ];
    var register = optionsAlias.registerOptions.bind(optionsAlias);

    optionsAlias.availableOptions = extendedAvailableOptions;
    expect(register).to.throw('The "carne-asada" alias is already in use by the "--filling" option and ' +
      'cannot be used by the "--favorite" option. Please use a different alias.');
  });

  it('registerOptions() should warn on options override attempts.', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var extendedAvailableOptions = [
      {
        name: 'spicy',
        type: Boolean,
        default: true,
        aliases: [
          { 'mild': true }
        ]
      }
    ];
    optionsAlias.registerOptions({ availableOptions: extendedAvailableOptions });
    expect(ui.output).to.match(/The ".*" alias cannot be overridden. Please use a different alias./);
  });

  it('registerOptions() should handle invalid alias definitions.', function() {
    //check for different types, validate proper errors are thrown
    var optionsAlias = new OptionsAliasCommand(options);
    var badArrayAvailableOptions = [{ name: 'filling', type: String, default: 'adobada', aliases: [
        'meat', [{ 'carne-asada': 'carne-asada' }], { 'carnitas': 'carnitas' }, { 'fish': 'fish' }
      ]
    }];
    var badObjectAvailableOptions = [{ name: 'filling', type: String, default: 'adobada', aliases: [
        'meat', { 'carne-asada': ['steak','grilled']}, { 'carnitas': 'carnitas' }, { 'fish': 'fish' }
      ]
    }];
    var register = optionsAlias.registerOptions.bind(optionsAlias);

    optionsAlias.availableOptions = badArrayAvailableOptions;
    expect(register).to.throw('The "[object Object]" [type:array] alias is not an acceptable value. ' +
      'It must be a string or single key object with a string value (for example, "value" or { "key" : "value" }).');

    optionsAlias.availableOptions = badObjectAvailableOptions;
    expect(register).to.throw('The "[object Object]" [type:object] alias is not an acceptable value. ' +
      'It must be a string or single key object with a string value (for example, "value" or { "key" : "value" }).');
  });

  it('parseAlias() should parse aliases and return an object', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var option = {
      name: 'filling',
      type: String,
      key: 'filling',
      default: 'adobada',
      aliases: [
        { 'carne-asada': 'carne-asada' },
        { 'carnitas': 'carnitas' },
        { 'fish': 'fish' }
      ]
    };
    var alias = { 'carnitas': 'carnitas' };
    expect(optionsAlias.parseAlias(option, alias)).to.deep.equal({
      key: 'carnitas',
      value: ['--filling','carnitas'],
      original: { 'carnitas': 'carnitas' }
    });
  });

  it('validateOption() should validate options', function() {
    var option = {
      name: 'filling',
      type: String,
      default: 'adobada',
      aliases: [
        { 'carne-asada': 'carne-asada' },
        { 'carnitas': 'carnitas' },
        { 'fish': 'fish' }
      ]
    };
    var dupe = { name: 'spicy', type: Boolean, default: true, aliases: [{ 'mild': false }] };
    var noAlias = { name: 'reload', type: Boolean, default: false };
    expect(new OptionsAliasCommand(options).validateOption(option)).to.be.ok;

    expect(new ServeCommand(options).validateOption(noAlias)).to.be.false;

    expect(new OptionsAliasCommand(options).validateOption(dupe)).to.be.false;
  });

  it('validateOption() should throw an error when option is missing name or type', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var notype = { name: 'taco' };
    var noname = { type: Boolean };

    expect(optionsAlias.validateOption.bind(optionsAlias, notype)).to.throw('The command "options-alias" has an ' +
      'option without the required type and name fields.');
    expect(optionsAlias.validateOption.bind(optionsAlias, noname)).to.throw('The command "options-alias" has an ' +
      'option without the required type and name fields.');
  });

  it('validateOption() should throw an error when option name is camelCase or capitalized', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var capital = {
      name: 'Taco',
      type: Boolean
    };
    var camel = {
      name: 'tacoTown',
      type: Boolean
    };

    expect(optionsAlias.validateOption.bind(optionsAlias, capital)).to.throw('The "Taco" option\'s name of the "options-alias"' +
      ' command contains a capital letter.');
    expect(optionsAlias.validateOption.bind(optionsAlias, camel)).to.throw('The "tacoTown" option\'s name of the "options-alias"' +
      ' command contains a capital letter.');
  });

  it('mergeDuplicateOption() should merge duplicate options together', function() {
    var optionsAlias = new OptionsAliasCommand(options);
    var garbageAvailableOptions = [
      { name: 'spicy', type: Boolean, default: true, aliases: [{ 'mild': true }] }
    ];
    optionsAlias.registerOptions({ availableOptions: garbageAvailableOptions });
    var extendedAvailableOptions = [{ name: 'filling', type: String, default: 'adobada', aliases: [
        { 'carne-asada': 'carne-asada' }, { 'carnitas': 'carnitas' }, { 'fish': 'fish' }
      ]
    }];
    var duplicateExtendedAvailableOptions = [{ name: 'filling', type: String, default: 'carnitas', aliases: [
        { 'pollo-asado': 'pollo-asado' }, { 'carne-asada': 'carne-asada' }
      ]
    }];
    optionsAlias.registerOptions({ availableOptions: extendedAvailableOptions });
    optionsAlias.availableOptions.push(duplicateExtendedAvailableOptions[0]);

    expect(optionsAlias.mergeDuplicateOption('filling')).to.deep.equal([
      {
        name: 'taco',
        type: String,
        default: 'traditional',
        aliases: [
          { 'hard-shell': 'hard-shell' },
          { 'soft-shell': 'soft-shell' }
        ],
        key: 'taco',
        required: false
      },
      {
        name: 'display-message',
        type: String,
        aliases: [
          'dm',
          { 'hw': 'Hello world' }
        ],
        key: 'displayMessage',
        required: false
      },
      {
        name: 'spicy',
        type: Boolean,
        default: true,
        aliases: [
          { 'mild': false }
        ],
        key: 'spicy',
        required: false
      },
      {
        name: 'filling',
        type: String,
        default: 'carnitas',
        aliases: [
          { 'carne-asada': 'carne-asada' },
          { 'carnitas': 'carnitas' },
          { 'fish': 'fish' },
          { 'pollo-asado': 'pollo-asado' }
        ],
        key: 'filling',
        required: false
      }
    ]);
  });

  it('implicit shorthands work with values.', function() {
    expect(new OptionsAliasCommand(options).parseArgs(['-s', 'false', '-t', 'hard-shell'])).to.deep.equal({
      options: {
        taco: 'hard-shell',
        spicy: false
      },
      args: []
    });
  });
});
