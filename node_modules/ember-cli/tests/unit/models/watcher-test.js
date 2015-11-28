'use strict';

var expect = require('chai').expect;

var MockUI = require('../../helpers/mock-ui');
var MockAnalytics = require('../../helpers/mock-analytics');
var MockWatcher  = require('../../helpers/mock-watcher');
var Watcher = require('../../../lib/models/watcher');
var EOL = require('os').EOL;
var chalk = require('chalk');
var BuildError = require('../../helpers/build-error');

describe('Watcher', function() {
  var ui;
  var subject;
  var builder;
  var analytics;
  var watcher;

  beforeEach(function() {
    ui        = new MockUI();
    analytics = new MockAnalytics();
    watcher   = new MockWatcher();

    subject = new Watcher({
      ui: ui,
      analytics: analytics,
      builder: builder,
      watcher: watcher
    });
  });

  describe('watcher strategy selection', function() {
    it('selects the events-based watcher by default', function () {
      subject.options = null;

      expect(subject.buildOptions()).to.deep.equal({
        verbose: true,
        poll: false,
        watchman: false,
        node: false
      });
    });

    it('selects the events-based watcher when given events watcher option', function () {
      subject.options = {
        watcher: 'events'
      };

      expect(subject.buildOptions()).to.deep.equal({
        verbose: true,
        poll: false,
        watchman: true,
        node: false
      });
    });

    it('selects the polling watcher when given polling watcher option', function () {
      subject.options = {
        watcher: 'polling'
      };

      expect(subject.buildOptions()).to.deep.equal({
        verbose: true,
        poll: true,
        watchman: false,
        node: false
      });
    });
  });

  describe('watcher:change', function() {
    beforeEach(function() {
      watcher.emit('change', {
        totalTime: 12344000000
      });
    });

    it('tracks events', function() {
      expect(analytics.tracks).to.deep.equal([{
        name: 'ember rebuild',
        message: 'broccoli rebuild time: 12344ms'
      }]);
    });

    it('tracks timings', function() {
      expect(analytics.trackTimings).to.deep.equal([{
        category: 'rebuild',
        variable: 'rebuild time',
        label:    'broccoli rebuild time',
        value:    12344
      }]);
    });

    it('logs that the build was successful', function() {
      expect(ui.output).to.equal(EOL + chalk.green('Build successful - 12344ms.') + EOL);
    });
  });

  describe('watcher:error', function() {
    it('tracks errors', function() {
      watcher.emit('error', {
        message: 'foo',
        stack: new Error().stack
      });

      expect(analytics.trackErrors).to.deep.equal([{
        description: 'foo'
      }]);
    });

    it('emits without error.file', function() {
      subject.didError(new BuildError({
        file: 'someFile',
        message: 'buildFailed'
      }));

      expect(ui.output).to.equal('');

      var outs = ui.errors.split(EOL);

      expect(outs[0]).to.equal(chalk.red('File: someFile'));
      expect(outs[1]).to.equal(chalk.red('buildFailed'));
    });

    it('emits with error.file with error.line without err.col', function() {
      subject.didError(new BuildError({
        file: 'someFile',
        line: 24,
        message: 'buildFailed'
      }));

      expect(ui.output).to.eql('');

      var outs = ui.errors.split(EOL);

      expect(outs[0]).to.equal(chalk.red('File: someFile (24)'));
      expect(outs[1]).to.equal(chalk.red('buildFailed'));
    });

    it('emits with error.file without error.line with err.col', function() {
      subject.didError(new BuildError({
        file: 'someFile',
        col: 80,
        message: 'buildFailed'
      }));

      expect(ui.output).to.eql('');

      var outs = ui.errors.split(EOL);

      expect(outs[0]).to.equal(chalk.red('File: someFile'));
      expect(outs[1]).to.equal(chalk.red('buildFailed'));
    });

    it('emits with error.file with error.line with err.col', function() {
      subject.didError(new BuildError({
        file: 'someFile',
        line: 24,
        col: 80,
        message: 'buildFailed'
      }));

      expect(ui.output).to.eql('');

      var outs = ui.errors.split(EOL);

      expect(outs[0]).to.equal(chalk.red('File: someFile (24:80)'));
      expect(outs[1]).to.equal(chalk.red('buildFailed'));
    });
  });

  describe('watcher:change afterError', function() {
    beforeEach(function() {
      watcher.emit('error', {
        message: 'foo',
        stack: new Error().stack
      });

      watcher.emit('change', {
        totalTime: 12344000000
      });
    });

    it('log that the build was green', function() {
      expect(ui.output).to.match(/Build successful./, 'has successful build output');
    });

    it('keep tracking analytics', function() {
      expect(analytics.tracks).to.deep.equal([{
        name: 'ember rebuild',
        message: 'broccoli rebuild time: 12344ms'
      }]);
    });
  });
});
