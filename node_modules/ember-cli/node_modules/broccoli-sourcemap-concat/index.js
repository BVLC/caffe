var Concat = require('./concat');
var merge = require('lodash-node/modern/objects/merge');

module.exports = function(inputNode, options) {
  if (!options || !options.outputFile) {
    throw new Error('outputFile is required');
  }

  var config = merge({
    enabled: true
  }, options.sourceMapConfig);

  var Strategy;

  if (config.enabled) {
    var extensions = (config.extensions || ['js']);
    for (var i=0; i < extensions.length; i++) {
      var ext = '.' + extensions[i].replace(/^\./,'');
      if (options.outputFile.slice(-1 * ext.length) === ext) {
        Strategy = require('fast-sourcemap-concat');
        break;
      }
    }
  }

  return new Concat(inputNode, options, Strategy || require('./concat-without-source-maps'));
};
