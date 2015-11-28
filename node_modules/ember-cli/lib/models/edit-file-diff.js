'use strict';

var fs           = require('fs');
var Promise      = require('../ext/promise');
var readFile     = Promise.denodeify(fs.readFile);
var writeFile    = Promise.denodeify(fs.writeFile);
var childProcess = require('child_process');
var jsdiff       = require('diff');
var quickTemp    = require('quick-temp');
var path         = require('path');
var SilentError  = require('silent-error');

function EditFileDiff(options) {
  this.info = options.info;

  quickTemp.makeOrRemake(this, 'tmpDifferenceDir');
}

EditFileDiff.prototype.edit = function(){
  return Promise.hash({
    input:  this.info.render(),
    output: readFile(this.info.outputPath)
  })
    .then(invokeEditor.bind(this))
    .then(applyPatch.bind(this))
    .finally(cleanUp.bind(this));
};

function cleanUp() {
  quickTemp.remove(this, 'tmpDifferenceDir'); // jshint ignore:line
}

function applyPatch(resultHash) {
  /*jshint validthis:true */
  return Promise.hash({
    diffString: readFile(resultHash.diffPath),
    currentString: readFile(resultHash.outputPath)
  }).then(function(result){
    var appliedDiff = jsdiff.applyPatch(result.currentString.toString(), result.diffString.toString());

    if(!appliedDiff) {
      var message = 'Patch was not cleanly applied.';
      this.info.ui.writeLine(message + ' Please choose another action.');
      throw new SilentError(message);
    }

    return writeFile(resultHash.outputPath, appliedDiff);
  }.bind(this));
}

function invokeEditor(result) {
  var info     = this.info; // jshint ignore:line
  var diff     = jsdiff.createPatch(info.outputPath, result.output.toString(), result.input);
  var diffPath = path.join(this.tmpDifferenceDir, 'currentDiff.diff'); // jshint ignore:line

  return new Promise(function(resolve, reject) {
    writeFile(diffPath, diff).then(function() {
      var editorArgs  = process.env.EDITOR.split(' ');
      var editor      = editorArgs.shift();
      var editProcess = childProcess.spawn(editor, [diffPath].concat(editorArgs), {stdio: 'inherit'});
      var results     = { outputPath: info.outputPath, diffPath: diffPath };

      editProcess.on('close', function(code){
        if (code === 0) {
          resolve(results);
        } else {
          reject();
        }
      });
    });
  });
}

module.exports = EditFileDiff;
