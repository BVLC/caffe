'use strict';

var expect         = require('chai').expect;
var merge          = require('lodash/object/merge');
var MockUI         = require('../../helpers/mock-ui');
var MockAnalytics  = require('../../helpers/mock-analytics');
var Command        = require('../../../lib/models/command');
var Yam            = require('yam');

describe('.ember-cli', function() {
  var ui;
  var analytics;
  var project;
  var settings;
  var homeSettings;

  before(function() {
    ui        = new MockUI();
    analytics = new MockAnalytics();
    project   = { isEmberCLIProject: function() { return true; }};

    homeSettings = {
      proxy:       'http://iamstef.net/ember-cli',
      liveReload:  false,
      environment: 'mock-development',
      host:        '0.1.0.1'
    };

    settings = new Yam('ember-cli', {
      secondary: process.cwd() + '/tests/fixtures/home',
      primary:   process.cwd() + '/tests/fixtures/project'
    }).getAll();
  });

  it('local settings take precendence over global settings', function() {
    var command = new Command({
      ui:        ui,
      analytics: analytics,
      project:   project,
      settings:  settings
    });

    var args = command.parseArgs();

    expect(args.options).to.include(
      merge(homeSettings, {
        port:       999,
        liveReload: false
      })
    );
  });
});

