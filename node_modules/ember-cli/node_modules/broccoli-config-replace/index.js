'use strict';

var fs = require('fs-extra'),
    path = require('path'),
    Plugin = require('broccoli-plugin'),
    hashStrings = require('broccoli-kitchen-sink-helpers').hashStrings;

function ConfigReplace(inputNode, configNode, options) {
  options = options || {};

  // this._super();
  Plugin.call(this, [inputNode, configNode], {
    annotation: options.annotation
  });

  this._persistentOutput = true;

  this.options = options;
  this._cache = {};
}

ConfigReplace.prototype = Object.create(Plugin.prototype);
ConfigReplace.prototype.constructor = ConfigReplace;

ConfigReplace.prototype.build = function () {
  var config = this.getConfig();

  this.options.files.forEach(function(file) {
    var key = this.deriveCacheKey(file),
        entry = this._cache[key.hash],
        contents, filePath;

    if (entry) { return; }

    filePath = file;
    contents = this.processFile(config, filePath);
    this.updateCache(key, contents);

    filePath = path.join(this.outputPath, filePath);
    this.writeFile(filePath, contents);
  }, this);
};

ConfigReplace.prototype.deriveCacheKey = function(file) {
  var stats = fs.statSync(this.getConfigPath());

  if (stats.isDirectory()) {
    throw new Error('Must provide a path for the config file, you supplied a directory');
  }

  var patterns = this.options.patterns.reduce(function(a, b) {
    return a + b.match;
  }, '');

  return {
    file: file,
    hash: hashStrings([
      file,
      this.configPath,
      patterns,
      stats.size,
      stats.mode,
      stats.mtime.getTime()
    ])
  };
};

ConfigReplace.prototype.processFile = function(config, filePath) {
  filePath = path.join(this.inputPaths[0], filePath);
  var contents = fs.readFileSync(filePath, { encoding: 'utf8' });

  this.options.patterns.forEach(function(pattern) {
    var replacement = pattern.replacement;

    if (typeof replacement === 'function') {
      replacement = function() {
        var args = Array.prototype.slice.call(arguments);
        return pattern.replacement.apply(null, [config].concat(args));
      }
    }

    contents = contents.replace(pattern.match, replacement);
  });

  return contents;
};

ConfigReplace.prototype.writeFile = function(destPath, contents) {
  if (!fs.existsSync(path.dirname(destPath))) {
    fs.mkdirSync(path.dirname(destPath));
  }

  fs.writeFileSync(destPath, contents, { encoding: 'utf8' });
};

ConfigReplace.prototype.updateCache = function(key, contents) {
  Object.keys(this._cache).forEach(function(hash) {
    if (this._cache[hash].file === key.file) {
      delete this._cache[hash];
    }
  }, this);

  this._cache[key.hash] = {
    file: key.file,
    contents: contents
  };
};

ConfigReplace.prototype.getConfigPath = function() {
  return path.join(this.inputPaths[1], this.options.configPath);
};

ConfigReplace.prototype.getConfig = function() {
  var contents = fs.readFileSync(this.getConfigPath(), { encoding: 'utf8' });
  return JSON.parse(contents);
};

module.exports = ConfigReplace;
