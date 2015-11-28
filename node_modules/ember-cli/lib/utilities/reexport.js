'use strict';

var fs = require('fs-extra');
var path = require('path');
var template = fs.readFileSync(path.join(__dirname, 'reexport-template.js'), 'utf-8');
var hashStrings = require('broccoli-kitchen-sink-helpers').hashStrings;
var Plugin = require('broccoli-plugin');

function Reexporter(name, outputFile, options) {
  options = options || {};
  Plugin.call(this, [], { persistentOutput: true, annotation: options.annotation });
  this.name = name;
  this.outputFile = outputFile;
  this.lastHash = undefined;
}
Reexporter.prototype = Object.create(Plugin.prototype);
Reexporter.prototype.constructor = Reexporter;

Reexporter.prototype.content = function() {
  return template
    .replace(/\s*\/\*.*\*\/\s*/, '')
    .replace('{{DEST}}', this.name)
    .replace('{{SRC}}', this.name + '/index');
};

Reexporter.prototype.build = function() {
  if (!this.subdirCreated) {
    fs.mkdirSync(path.join(this.outputPath, 'reexports'));
    this.subdirCreated = true;
  }

  var outputPath = path.join(this.outputPath, 'reexports', this.outputFile);

  var content = this.content();
  var hash = hashStrings([content]);

  if (this.lastHash !== hash) {
    this.lastHash = hash;
    fs.writeFileSync(outputPath, content);
  }
};

module.exports = Reexporter;
