'use strict';

var expect         = require('chai').expect;
var commandOptions = require('../../factories/command-options');
var NewCommand     = require('../../../lib/commands/new');

describe('new command', function() {
  var command;

  beforeEach(function() {
    var options = commandOptions({
      project: {
        isEmberCLIProject: function() {
          return false;
        }
      }
    });

    command = new NewCommand(options);
  });

  it('doesn\'t allow to create an application named `test`', function() {
    return command.validateAndRun(['test']).then(function() {
      expect(false, 'should have rejected with an application name of test');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `test`.');
    });
  });

  it('doesn\'t allow to create an application named `ember`', function() {
    return command.validateAndRun(['ember']).then(function() {
      expect(false, 'should have rejected with an application name of ember');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `ember`.');
    });
  });

  it('doesn\'t allow to create an application named `Ember`', function() {
    return command.validateAndRun(['Ember']).then(function() {
      expect(false, 'should have rejected with an application name of Ember');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `Ember`.');
    });
  });

  it('doesn\'t allow to create an application named `ember-cli`', function() {
    return command.validateAndRun(['ember-cli']).then(function() {
      expect(false, 'should have rejected with an application name of ember-cli');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `ember-cli`.');
    });
  });

  it('doesn\'t allow to create an application named `vendor`', function() {
    return command.validateAndRun(['vendor']).then(function() {
      expect(false, 'should have rejected with an application name of `vendor`');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `vendor`.');
    });
  });

  it('doesn\'t allow to create an application with a period in the name', function() {
    return command.validateAndRun(['zomg.awesome']).then(function() {
      expect(false, 'should have rejected with period in the application name');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `zomg.awesome`.');
    });
  });

  it('doesn\'t allow to create an application with a name beginning with a number', function() {
    return command.validateAndRun(['123-my-bagel']).then(function() {
      expect(false, 'should have rejected with a name beginning with a number');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `123-my-bagel`.');
    });
  });

  it('shows a suggestion messages when the application name is a period', function() {
    return command.validateAndRun(['.']).then(function() {
      expect(false, 'should have rejected with a name `.`');
    })
    .catch(function(error) {
      expect(error.message).to.equal('Trying to generate an application structure in this directory? Use `ember init` instead.');
    });
  });
});
