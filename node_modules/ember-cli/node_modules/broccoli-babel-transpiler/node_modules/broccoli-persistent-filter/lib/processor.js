'use strict';

module.exports = Processor;

function Processor(options) {
  options = options || {};
  this.processor = {};
  this.persistent = options.persist;
}

Processor.prototype.setStrategy = function(stringProcessor) {
  this.processor = stringProcessor;
};

Processor.prototype.init = function(ctx) {
  this.processor.init(ctx);
};

Processor.prototype.processString = function(ctx, contents, relativePath) {
  return this.processor.processString(ctx, contents, relativePath);
};
