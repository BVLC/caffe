'use strict';

var expect         = require('chai').expect;
var commandOptions = require('../../factories/command-options');
var AddonCommand   = require('../../../lib/commands/addon');

describe('addon command', function() {
  var command;

  beforeEach(function() {
    var options = commandOptions({
      project: {
        isEmberCLIProject: function() {
          return false;
        }
      }
    });

    command = new AddonCommand(options);
  });

  it('doesn\'t allow to create an addon named `test`', function() {
    return command.validateAndRun(['test']).then(function() {
      expect(false, 'should have rejected with an addon name of test');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `test`.');
    });
  });

  it('doesn\'t allow to create an addon named `ember`', function() {
    return command.validateAndRun(['ember']).then(function() {
      expect(false, 'should have rejected with an addon name of test');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `ember`.');
    });
  });

  it('doesn\'t allow to create an addon named `vendor`', function() {
    return command.validateAndRun(['vendor']).then(function() {
      expect(false, 'should have rejected with an addon name of `vendor`');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `vendor`.');
    });
  });

  it('doesn\'t allow to create an addon with a period in the name', function() {
    return command.validateAndRun(['zomg.awesome']).then(function() {
      expect(false, 'should have rejected with period in the addon name');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `zomg.awesome`.');
    });
  });

  it('doesn\'t allow to create an addon with a name beginning with a number', function() {
    return command.validateAndRun(['123-my-bagel']).then(function() {
      expect(false, 'should have rejected with a name beginning with a number');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `123-my-bagel`.');
    });
  });
});
