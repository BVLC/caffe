/*
 * index.js: Top-level plugin exposing CLI features in flatiron
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var fs = require('fs'),
    path = require('path'),
    flatiron = require('../../flatiron'),
    common = flatiron.common,
    director = require('director');

//
// ### Name this plugin
//
exports.name = 'cli';

//
// ### function attach (options, done)
// #### @options {Object} Options for this plugin
// Initializes `this` (the application) with the core `cli` plugins consisting of:
// `argv`, `prompt`, `routing`, `commands` in that order.
//
exports.attach = function (options) {
  var app = this;
  options = options || {};

  //
  // Define the `cli` namespace on the app for later use
  //
  app.cli = app.cli || {};

  //
  // Mixin some keys properly so that plugins can also set them
  //
  options.argv = common.mixin({}, app.cli.argv || {}, options.argv || {});
  options.prompt = common.mixin({}, app.cli.prompt || {}, options.prompt || {});

  app.cli = common.mixin({}, app.cli, options);

  if (app.cli.notFoundUsage == undefined) {
    app.cli.notFoundUsage = true;
  }

  //
  // Setup `this.argv` to use `optimist`.
  //
  exports.argv.call(this, app.cli.argv);
  app.use(flatiron.plugins.inspect);

  //
  // If `options.version` is truthy, `app.version` is defined and `-v` or
  // `--version` command line parameters were passed, print out `app.version`
  // and exit.
  //
  if (app.cli.version && app.version && (this.argv.v || this.argv.version)) {
    console.log(app.version);
    process.exit(0);
  }

  //
  // Setup `this.prompt`.
  //
  exports.prompt.call(this, app.cli.prompt);

  //
  // Setup `app.router` and associated core routing method.
  //
  app.router = new director.cli.Router().configure({
    async: app.async || app.cli.async
  });

  app.start = function (options, callback) {
    if (!callback && typeof options === 'function') {
      callback = options;
      options = {};
    }

    callback = callback || function () {};
    app.init(options, function (err) {
      if (err) {
        return callback(err);
      }

      app.router.dispatch('on', app.argv._.join(' '), app.log, callback);
    });
  };

  app.cmd = function (path, handler) {
    app.router.on(path, handler);
  };

  exports.commands.call(this);
};

//
// ### function init (done)
// #### @done {function} Continuation to respond to when complete
// Initializes this plugin by setting `winston.cli` (i.e. `app.log.cli`)
// to enable colors and padded levels.
//
exports.init = function (done) {
  var app = this,
      logger;

  if (!app.log.help) {
    logger = app.log.get('default');
    logger.cli().extend(app.log);
  }

  if (app.config) {
    //
    // Create a literal store for argv to
    // avoid re-parsing CLI options.
    //
    app.config.use('argv', {
      type: 'literal',
      store: app.argv
    });

    app.config.env();
  }

  done();
};

//
// ### function argv (options)
// #### @options {Object} Pass-thru options for optimist
// Sets up `app.argv` using `optimist` and the specified options.
//
exports.argv = function (options) {
  var optimist = require('optimist').string('_');

  if (options && Object.keys(options).length) {
    optimist = optimist.options(options);
    this.showOptions = optimist.help;
    this.argv = optimist.argv;
  }
  else {
    this.showOptions = optimist.help;
    this.argv = optimist.argv;
  }
};

//
// ### function commands (options)
// #### @options {Object} Options for the application commands
// Configures the `app.commands` object which is lazy-loaded from disk
// along with some default logic for: `help` and `alias`.
//
exports.commands = function (options) {
  var app = this;

  function showUsage(target) {
    target = Array.isArray(target) ? target : target.split('\n');
    target.forEach(function (line) {
      app.log.help(line);
    });

    var lines = app.showOptions().split('\n').filter(Boolean);

    if (lines.length) {
      app.log.help('');
      lines.forEach(function (line) {
        app.log.help(line);
      });
    }
  }

  //
  // Setup any pass-thru options to the
  // application instance but make them lazy
  //
  app.usage = app.cli.usage;
  app.cli.source = app.cli.dir || app.cli.source;
  app.commands = app.commands || {};

  //
  // Helper function which loads the file for the
  // specified `name` into `app.commands`.
  //
  function loadCommand(name, command, silent) {
    var resource = app.commands[name];
    var usage = app.usage || [
      name
        ? 'Cannot find commands for ' + name.magenta
        : 'Cannot find commands'
    ];

    if (resource && (!command || resource[command])) {
      return true;
    }

    if (app.cli.source) {
      if (!app.cli.sourceDir) {
        try {
          var stats = fs.statSync(app.cli.source);
          app.cli.sourceDir = stats.isDirectory();
        }
        catch (ex) {
          if (app.cli.notFoundUsage) {
            showUsage(usage)
          }

          return false;
        }
      }

      try {
        if (app.cli.sourceDir) {
          app.commands[name] = require(path.join(app.cli.source, name || ''));
        }
        else {
          app.commands = common.mixin(app.commands, require(app.cli.source));
        }
        return true;
      }
      catch (err) {
        // If that file could not be found, error message should start with
        // "Cannot find module" and contain the name of the file we tried requiring.
        if (!err.message.match(/^Cannot find module/) || (name && err.message.indexOf(name) === -1)) {
          throw err;
        }

        if (!silent) {
          if (app.cli.notFoundUsage) {
            showUsage(usage);
          }
        }

        return false;
      }
    }
  }

  //
  // Helper function to ensure the user wishes to execute
  // a destructive command.
  //
  function ensureDestroy(callback) {
    app.prompt.get(['destroy'], function (err, result) {
      if (result.destroy !== 'yes' && result.destroy !== 'y') {
        app.log.warn('Destructive operation cancelled');
        return callback(true);
      }

      callback();
    });
  }

  //
  // Helper function which executes the command
  // represented by the Array of `parts` passing
  // control to the `callback`.
  //
  function executeCommand(parts, callback) {
    var name,
        shouldLoad = true,
        command,
        usage;

    if (typeof parts === 'undefined' || typeof parts === 'function') {
      throw(new Error('parts is a required argument of type Array'));
    }

    name = parts.shift();

    if (app.cli.source || app.commands[name]) {
      if (app.commands[name]) {
        shouldLoad = false;
        if (typeof app.commands[name] != 'function' && !app.commands[name][parts[0]]) {
          shouldLoad = true;
        }
      }

      if (shouldLoad && !loadCommand(name, parts[0])) {
        return callback();
      }

      command = app.commands[name];
      while (command) {
        usage = command.usage;

        if (!app.argv.h && !app.argv.help && typeof command === 'function') {
          while (parts.length + 1 < command.length) {
            parts.push(null);
          }

          if (command.destructive) {
            return ensureDestroy(function (err) {
              return err ? callback() : command.apply(app, parts.concat(callback));
            })
          }

          command.apply(app, parts.concat(callback));
          return;
        }

        command = command[parts.shift()];
      }

      //
      // Since we have not resolved a needle, try and print out a usage message
      //
      if (usage || app.cli.usage) {
        showUsage(usage || app.cli.usage);
        callback(false);
      }
    }
    else if (app.usage) {
      //
      // If there's no directory we're supposed to search for modules, simply
      // print out usage notice if it's provided.
      //
      showUsage(app.cli.usage);
      callback(true);
    }
  }

  //
  // Expose the executeCommand method
  //
  exports.executeCommand = executeCommand;

  //
  // Allow commands to be aliased to subcomponents. e.g.
  //
  //    app.alias('list', { resource: 'apps', command: 'list' });
  //    app.alias('new', { command: 'create' });
  //    app.alias('new', 'create');
  //
  app.alias = function (target, source) {
    app.commands.__defineGetter__(target, function () {

      var resource = source.resource || source.command || source,
          command = source.resource ? source.command : null;

      loadCommand(resource, command, true);
      resource = app.commands[resource];

      if (resource) {
        return source.resource && source.command
          ? resource[source.command]
          : resource;
      }
    });
  };

  //
  // Set the `loadCommand` function to run
  // whenever the router has not matched
  // the CLI arguments, `process.argv`.
  //
  app.router.notfound = function (callback) {
    executeCommand(app.argv._.slice(), callback);
  };

  //
  // Setup default help command
  //
  app.cmd(/help ([^\s]+)?\s?([^\s]+)?/, app.showHelp = function showHelp() {
    var args = Array.prototype.slice.call(arguments).filter(Boolean),
        callback = typeof args[args.length - 1] === 'function' && args.pop(),
        resource,
        usage;

    function displayAndRespond(found) {
      showUsage(usage || app.usage);
      if (!found) {
        app.log.warn('Cannot find help for ' + args.join(' ').magenta);
      }

      if (callback) {
        callback();
      }
    }

    if (!loadCommand(args[0], args[1], true)) {
      return displayAndRespond(false);
    }

    resource = app.commands[args[0]];
    usage = resource.usage;

    for (var i = 1; i < args.length; i++) {
      if (!resource[args[i]]) {
        return displayAndRespond(false);
      }
      else if (resource[args[i]].usage) {
        resource = resource[args[i]];
        usage = resource.usage;
      }
    }

    displayAndRespond(true);
  });
};

//
// ### function prompt (options)
// #### @options {Object} Options for the prompt.
// Sets up the application `prompt` property to be a lazy
// setting which loads the `prompt` module.
//
exports.prompt = function (options) {
  options = options || {};

  this.__defineGetter__('prompt', function () {
    if (!this._prompt) {
      //
      // Pass-thru any prompt specific options that are supplied.
      //
      var prompt = require('prompt'),
          self = this;

      prompt.allowEmpty = options.allowEmpty || prompt.allowEmpty;
      prompt.message    = options.message    || prompt.message;
      prompt.delimiter  = options.delimiter  || prompt.delimiter;
      prompt.properties = options.properties || prompt.properties;

      //
      // Setup `destroy` property for destructive commands
      //
      prompt.properties.destroy = {
        name: 'destroy',
        message: 'This operation cannot be undone, Would you like to proceed?',
        default: 'yes'
      };

      //
      // Hoist up any prompt specific events and re-emit them as
      // `prompt::*` events.
      //
      ['start', 'pause', 'resume', 'prompt', 'invalid'].forEach(function (ev) {
        prompt.on(ev, function () {
          var args = Array.prototype.slice.call(arguments);
          self.emit.apply(self, [['prompt', ev]].concat(args));
        });
      });

      //
      // Extend `this` (the application) with prompt functionality
      // and open `stdin`.
      //
      this._prompt = prompt;
      this._prompt.start().pause();
    }

    return this._prompt;
  });
};
