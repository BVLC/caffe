'use strict';

var fs = require('fs'),
    path = require('path'),
    CachingWriter = require('broccoli-caching-writer');

function ConfigLoader(inputNode, options) {
  options = options || {};
  CachingWriter.call(this, [inputNode], { annotation: options.annotation });
  this.options = options;
}

ConfigLoader.prototype = Object.create(CachingWriter.prototype);
ConfigLoader.prototype.constructor = ConfigLoader;

/*
 * @private
 *
 * On windows, when residing on a UNC share (lib or app/addon code), exact
 * match here is not possible. Although we could be more precise, there is
 * little pain in evicting all fuzzy matches
 *
 * @method fuzzyPurgeRequireEntry
 */

function fuzzyPurgeRequireEntry(entry) {
  return Object.keys(require.cache).filter(function(path) {
    return path.indexOf(entry) > -1;
  }).forEach(function(entry) {
    delete require.cache[entry];
  });
}

ConfigLoader.prototype.clearConfigGeneratorCache = function() {
  fuzzyPurgeRequireEntry(this.options.project.configPath() + '.js');
};

ConfigLoader.prototype.build = function() {
  this.clearConfigGeneratorCache();

  var outputDir = path.join(this.outputPath, 'environments');
  fs.mkdirSync(outputDir);

  var environments = [this.options.env];
  if (this.options.tests) {
    environments.push('test');
  }

  environments.forEach(function(env) {
    var config = this.options.project.config(env),
        jsonString = JSON.stringify(config),
        outputPath = path.join(outputDir, env);

    fs.writeFileSync(outputPath + '.json', jsonString, {
      encoding: 'utf8'
    });
  }, this);
};

module.exports = ConfigLoader;
