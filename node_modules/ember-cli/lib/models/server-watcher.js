'use strict';

var Task    = require('./task');

module.exports = Task.extend({
  verbose: true,

  init: function() {
    this.watcher = this.watcher || new (require('sane'))(this.watchedDir, {
      verbose: this.verbose,
      poll: this.polling()
    });

    this.watcher.on('change', this.didChange.bind(this));
    this.watcher.on('add',    this.didAdd.bind(this));
    this.watcher.on('delete', this.didDelete.bind(this));
  },

  didChange: function (filepath) {
    this.ui.writeLine('Server file changed: ' + filepath);

    this.analytics.track({
      name: 'server file change',
      description: 'File changed: "' + filepath + '"'
    });
  },

  didAdd: function (filepath) {
    this.ui.writeLine('Server file added: ' + filepath);

    this.analytics.track({
      name: 'server file addition',
      description: 'File added: "' + filepath + '"'
    });
  },

  didDelete: function (filepath) {
    this.ui.writeLine('Server file deleted: ' + filepath);

    this.analytics.track({
      name: 'server file deletion',
      description: 'File deleted: "' + filepath + '"'
    });
  },

  then: function() {
    return this.watcher.then.apply(this.watcher, arguments);
  },

  on: function() {
    this.watcher.on.apply(this.watcher, arguments);
  },

  off: function() {
    this.watcher.off.apply(this.watcher, arguments);
  },

  polling: function () {
    return this.options && this.options.watcher === 'polling';
  }
});
