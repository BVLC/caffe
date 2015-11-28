'use strict';

var fs          = require('fs-extra');
var existsSync  = require('exists-sync');
var path        = require('path');
var Promise     = require('../ext/promise');
var remove      = Promise.denodeify(fs.remove);
var Task        = require('./task');
var SilentError = require('silent-error');
var chalk       = require('chalk');
var cpd         = require('ember-cli-copy-dereference');
var attemptNeverIndex = require('../utilities/attempt-never-index');
var deprecate   = require('../utilities/deprecate');
var findBuildFile = require('../utilities/find-build-file');
var viz = require('broccoli-viz');
var FSMonitor = require('fs-monitor-stack');

var signalsTrapped = false;
var buildCount = 0;

function outputViz(count, result, monitor) {
  var processed = viz.process(result.graph);

  processed.forEach(function(node) {
    node.stats.fs = monitor.statsFor(node);
  });

  fs.writeFileSync('graph.' + count + '.dot', viz.dot(processed));
  fs.writeFileSync('graph.' + count + '.json', JSON.stringify({
    summary: {
      buildCount: count,
      output: result.directory,
      totalTime: result.totalTime,
      totalNodes: processed.length,
      stats: {
        fs: monitor.totalStats()
      }
    },
    nodes: processed
  }));
}

module.exports = Task.extend({
  setupBroccoliBuilder: function() {
    this.environment = this.environment || 'development';
    process.env.EMBER_ENV = process.env.EMBER_ENV || this.environment;

    var broccoli = require('broccoli');
    var hasBrocfile = existsSync(path.join('.', 'Brocfile.js'));
    var buildFile = findBuildFile('ember-cli-build.js');

    deprecate('Brocfile.js has been deprecated in favor of ember-cli-build.js. Please see the transition guide: https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md#user-content-brocfile-transition.', hasBrocfile);

    if (hasBrocfile) {
      this.tree = broccoli.loadBrocfile();
    } else if (buildFile) {
      this.tree = buildFile({ project: this.project });
    } else {
      throw new Error('No ember-cli-build.js found. Please see the transition guide: https://github.com/ember-cli/ember-cli/blob/master/TRANSITION.md#user-content-brocfile-transition.');
    }

    this.builder = new broccoli.Builder(this.tree);

    var builder = this;

    if (process.env.BROCCOLI_VIZ) {
      this.builder.on('start', function() {
        builder.monitor = new FSMonitor();
      });

      this.builder.on('nodeStart', function(node) {
        builder.monitor.push(node);
      });

      this.builder.on('nodeEnd', function() {
        builder.monitor.pop();
      });
    }
  },

  trapSignals: function() {
    if (!signalsTrapped) {
      process.on('SIGINT',  this.onSIGINT.bind(this));
      process.on('SIGTERM', this.onSIGTERM.bind(this));
      process.on('message', this.onMessage.bind(this));
      signalsTrapped = true;
    }
  },

  init: function() {
    this.setupBroccoliBuilder();
    this.trapSignals();
  },

  /**
    Determine whether the output path is safe to delete. If the outputPath
    appears anywhere in the parents of the project root, the build would
    delete the project directory. In this case return `false`, otherwise
    return `true`.
  */
  canDeleteOutputPath: function(outputPath) {
    var rootPathParents = [this.project.root];
    var dir = path.dirname(this.project.root);
    rootPathParents.push(dir);
    while (dir !== path.dirname(dir)) {
      dir = path.dirname(dir);
      rootPathParents.push(dir);
    }
    return rootPathParents.indexOf(outputPath) === -1;
  },

  /**
    This is used to ensure that the output path is emptied, but not deleted
    itself. If we simply used `remove(this.outputPath)`, any symlinks would
    now be broken. This iterates the direct children of the output path,
    and calls `remove` on each (this preserving any symlinks).
  */
  clearOutputPath: function() {
    var outputPath = this.outputPath;
    if (!existsSync(outputPath)) { return Promise.resolve();}

    if(!this.canDeleteOutputPath(outputPath)) {
      return Promise.reject(new SilentError('Using a build destination path of `' + outputPath + '` is not supported.'));
    }

    var promises = [];
    var entries = fs.readdirSync(outputPath);

    for (var i = 0, l = entries.length; i < l; i++) {
      promises.push(remove(path.join(outputPath, entries[i])));
    }

    return Promise.all(promises);
  },

  copyToOutputPath: function(inputPath) {
    var outputPath = this.outputPath;

    return new Promise(function(resolve) {
      if (!existsSync(outputPath)) {
        fs.mkdirsSync(outputPath);
      }

      resolve(cpd.sync(inputPath, outputPath));
    });
  },

  processBuildResult: function(results) {
    var self = this;

    return this.clearOutputPath()
      .then(function() {
        return self.copyToOutputPath(results.directory);
      })
      .then(function() {
        return results;
      });
  },

  processAddonBuildSteps: function(buildStep, results) {
    var addonPromises = [];
    if (this.project && this.project.addons.length) {
      addonPromises = this.project.addons.map(function(addon){
        if (addon[buildStep]) {
          return addon[buildStep](results);
        }
      }).filter(Boolean);
    }

    return Promise.all(addonPromises).then(function() {
      return results;
    });
  },

  build: function() {
    var self = this;
    var args = [];
    for (var i = 0, l = arguments.length; i < l; i++) {
      args.push(arguments[i]);
    }

    attemptNeverIndex('tmp');

    return this.processAddonBuildSteps('preBuild')
       .then(function() {
         return self.builder.build.apply(self.builder, args);
       })
      .then(function(result) {
        if (process.env.BROCCOLI_VIZ) {
          outputViz(buildCount++, result, self.monitor);
        }
        return result;
      })
      .then(this.processAddonBuildSteps.bind(this, 'postBuild'))
      .then(this.processBuildResult.bind(this))
      .then(this.processAddonBuildSteps.bind(this, 'outputReady'))
      .catch(function(error) {
        this.processAddonBuildSteps('buildError', error);
        throw error;
      }.bind(this));
  },

  cleanup: function() {
    var ui = this.ui;

    return this.builder.cleanup().catch(function(err) {
      ui.writeLine(chalk.red('Cleanup error.'));
      ui.writeError(err);
    });
  },

  cleanupAndExit: function() {
    this.cleanup().finally(function() {
      process.exit(1);
    });
  },

  onSIGINT: function() {
    this.cleanupAndExit();
  },
  onSIGTERM: function() {
    this.cleanupAndExit();
  },
  onMessage: function(message) {
    if (message.kill) {
      this.cleanupAndExit();
    }
  }
});
