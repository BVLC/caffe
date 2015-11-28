'use strict';

var transpiler = require('babel-core');
var Filter     = require('broccoli-persistent-filter');
var clone      = require('clone');
var path       = require('path');
var fs         = require('fs');
var stringify  = require('json-stable-stringify');
var mergeTrees = require('broccoli-merge-trees');
var funnel     = require('broccoli-funnel');
var crypto     = require('crypto');

function getExtensionsRegex(extensions) {
  return extensions.map(function(extension) {
    return new RegExp('\.' + extension + '$');
  });
}

function replaceExtensions(extensionsRegex, name) {
  for (var i = 0, l = extensionsRegex.length; i < l; i++) {
    name = name.replace(extensionsRegex[i], '');
  }

  return name;
}

function Babel(inputTree, _options) {
  if (!(this instanceof Babel)) {
    return new Babel(inputTree, _options);
  }

  var options = _options || {};
  options.persist = !options.exportModuleMetadata; // TODO: make this also work in cache
  Filter.call(this, inputTree, options);

  delete options.persist;
  this.options = options;
  this.moduleMetadata = {};
  this.extensions = this.options.filterExtensions || ['js'];
  this.extensionsRegex = getExtensionsRegex(this.extensions);
  this.name = 'broccoli-babel-transpiler';

  if (this.options.exportModuleMetadata) {
    this.exportModuleMetadata = this.options.exportModuleMetadata;
  }
  // Note, Babel does not support this option so we must save it then
  // delete it from the options hash
  delete this.options.exportModuleMetadata;

  if (this.options.browserPolyfill) {
    var babelCorePath = require.resolve('babel-core');
    babelCorePath = babelCorePath.replace(/\/babel-core\/.*$/, '/babel-core');

    var polyfill = funnel(babelCorePath, { files: ['browser-polyfill.js'] });
    this.inputTree = mergeTrees([polyfill, inputTree]);
  } else {
    this.inputTree = inputTree;
  }
  delete this.options.browserPolyfill;
}

Babel.prototype = Object.create(Filter.prototype);
Babel.prototype.constructor = Babel;
Babel.prototype.targetExtension = ['js'];

Babel.prototype.baseDir = function() {
  return __dirname;
};

Babel.prototype.build = function() {
  var self = this;
  return Filter.prototype.build.call(this).then(function() {
    if (self.exportModuleMetadata) {
      self._generateDepGraph();
    }
  });
};

Babel.prototype._generateDepGraph = function() {
  var residentImports = this._cache.keys().map(byImportName);
  var imports = Object.keys(this.moduleMetadata);
  var evictedImports = diff(imports, residentImports);

  if (evictedImports.length > 0) {
    evictedImports.forEach(function(importName) {
      delete this.moduleMetadata[importName];
    }, this);
  }

  fs.writeFileSync(this.outputPath + path.sep + 'dep-graph.json', stringify(this.moduleMetadata, { space: 2 }));
};

Babel.prototype.transform = function(string, options) {
  return transpiler.transform(string, options);
};

/*
 * @private
 *
 * @method optionsString
 * @returns a stringifeid version of the input options
 */
Babel.prototype.optionsHash  = function() {
  if (!this._optionsHash) {
    this._optionsHash = crypto.createHash('md5').update(stringify(this.options), 'utf8').digest('hex');
  }
  return this._optionsHash;
};

Babel.prototype.cacheKeyProcessString = function(string, relativePath) {
  return this.optionsHash() + Filter.prototype.cacheKeyProcessString.call(this, string, relativePath);
};

Babel.prototype.processString = function (string, relativePath) {
  var options = this.copyOptions();

  options.filename = options.sourceMapName = options.sourceFileName = relativePath;

  if (options.moduleId === true) {
    options.moduleId = replaceExtensions(this.extensionsRegex, options.filename);
  }

  var transpiled = this.transform(string, options);
  var key = options.moduleId ? options.moduleId : relativePath;

  if (transpiled.metadata && transpiled.metadata.modules) {
    this.moduleMetadata[byImportName(key)] = transpiled.metadata.modules;
  }

  return transpiled.code;
};

Babel.prototype.copyOptions = function() {
  var cloned = clone(this.options);
  if (cloned.filterExtensions) {
    delete cloned.filterExtensions;
  }
  return cloned;
};

function byImportName(relativePath) {
  return relativePath.replace(path.extname(relativePath), '');
}

function diff(array, exclusions) {
  return array.filter(function(item) {
    return !exclusions.some(function(exclude) {
      return item === exclude;
    });
  });
}

module.exports = Babel;
