'use strict';

var Task        = require('../models/task');
var Promise     = require('../ext/promise');
var SilentError = require('silent-error');

module.exports = Task.extend({
  init: function() {
    this.testem = this.testem || new (require('testem'))();
  },
  invokeTestem: function (options) {
    var testem = this.testem;

    return new Promise(function(resolve, reject) {
      testem.startCI(this.testemOptions(options), function(exitCode) {
        if (!testem.app.reporter.total) {
          reject(new SilentError('No tests were run, please check whether any errors occurred in the page (ember test --server) and ensure that you have a test launcher (e.g. PhantomJS) enabled.'));
        }

        resolve(exitCode);
      });
    }.bind(this));
  },

  addonMiddlewares: function() {
    this.project.initializeAddons();

    return this.project.addons.reduce(function(addons, addon) {
      if (addon.testemMiddleware) {
        addons.push(addon.testemMiddleware.bind(addon));
      }

      return addons;
    }, []);
  },

  testemOptions: function(options) {
    var testemOptions = {
      host: options.host,
      port: options.port,
      cwd: options.outputPath,
      reporter: options.reporter,
      middleware: this.addonMiddlewares(),
      launch: options.launch,
      /* jshint ignore:start */
      config_dir: process.cwd(),
      test_page: options.testPage,
      query_params: options.queryString
      /* jshint ignore:end */
    };

    return testemOptions;
  },

  run: function(options) {
    return this.invokeTestem(options);
  }
});
